"""API tests. Trains a fresh model into a temp dir (a few seconds), so the served model in
models/ is never touched. Needs the Telco CSV at CHURN_DATA_PATH (CI downloads IBM's copy)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service import app as app_module
from service import churn_model, complaints, drift

LOW_RISK = dict(gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=70,
                PhoneService="Yes", MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes",
                OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
                StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
                PaymentMethod="Bank transfer (automatic)", MonthlyCharges=65.0, TotalCharges=4550.0)
HIGH_RISK = dict(LOW_RISK, Partner="No", Dependents="No", tenure=1, InternetService="Fiber optic",
                 OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                 Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check",
                 MonthlyCharges=95.0, TotalCharges=95.0)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    mp = pytest.MonkeyPatch()
    mp.setattr(churn_model, "MODEL_DIR", tmp)
    mp.setattr(churn_model, "PIPELINE_PATH", tmp / "pipeline.joblib")
    mp.setattr(churn_model, "META_PATH", tmp / "meta.json")
    mp.setattr(app_module, "ACTIONS_LOG_PATH", tmp / "actions.jsonl")
    mp.setattr(app_module, "API_KEY", None)
    app_module._state["model"] = None
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


def demo(client, **params):
    return client.get("/demo/customers", params=params).json()


def test_health_reports_trained_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["churn_model"]["cv_roc_auc_mean"] > 0.8


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_KEY", "k")
    assert client.post("/score", json={"customers": [LOW_RISK]}).status_code == 401
    ok = client.post("/score", json={"customers": [LOW_RISK]}, headers={"X-API-Key": "k"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for probes


def test_score_shape_and_explanations(client):
    customers = demo(client, n=20, seed=1)["customers"]
    preds = client.post("/score", json={"customers": customers}).json()["predictions"]
    assert [p["customer_id"] for p in preds] == [c["customerID"] for c in customers]
    for p in preds:
        assert 0.0 <= p["churn_probability"] <= 1.0
        assert len(p["top_risk_factors"]) <= 3
        for f in p["top_risk_factors"]:
            assert f["feature"] in churn_model.FEATURES and f["impact"] > 0


def test_risk_ordering_is_sensible(client):
    preds = client.post("/score", json={"customers": [LOW_RISK, HIGH_RISK]}).json()["predictions"]
    low, high = (p["churn_probability"] for p in preds)
    assert high > 0.5 > low
    assert "Contract" in [f["feature"] for f in preds[1]["top_risk_factors"]]


def test_probabilities_are_calibrated_on_average(client):
    customers = demo(client, n=2000, seed=3)["customers"]
    preds = client.post("/score", json={"customers": customers}).json()["predictions"]
    mean = np.mean([p["churn_probability"] for p in preds])
    assert abs(mean - 0.265) < 0.04  # Telco base churn rate; the raw model averages ~0.40


def test_score_validation(client):
    bad = {k: v for k, v in LOW_RISK.items() if k != "Contract"}
    assert client.post("/score", json={"customers": [bad]}).status_code == 422
    assert client.post("/score", json={"customers": []}).status_code == 422


@pytest.mark.parametrize("scenario,expect_retrain", [("none", False), ("price_hike", False),
                                                      ("contract_shift", True), ("both", True)])
def test_drift_check_scenarios(client, scenario, expect_retrain):
    d = demo(client, n=800, seed=5, scenario=scenario, include_labels="true")
    r = client.post("/drift/check", json={"customers": d["customers"], "churned": d["churned"]}).json()
    assert r["retrain_recommended"] is expect_retrain
    if scenario in ("contract_shift", "both"):
        assert "Contract" in r["drifted_features"]
    if scenario == "price_hike":
        # drifts, but in a feature the model barely relies on - the point of importance weighting
        assert "MonthlyCharges" in r["drifted_features"] and r["drifted_importance_mass"] < 0.3
    assert r["current_roc_auc"] is not None


def test_drift_check_rejects_small_or_mismatched_batches(client):
    d = demo(client, n=200, seed=6, include_labels="true")
    assert client.post("/drift/check", json={"customers": d["customers"][:50]}).status_code == 422
    r = client.post("/drift/check", json={"customers": d["customers"], "churned": d["churned"][:10]})
    assert r.status_code == 422


def test_retrain_promotion_gate(client, monkeypatch):
    before = client.get("/health").json()["churn_model"]["trained_at"]
    monkeypatch.setattr(app_module, "PROMOTION_TOLERANCE", -0.05)  # demand a +0.05 AUC gain
    r = client.post("/retrain").json()
    assert r["promoted"] is False
    assert client.get("/health").json()["churn_model"]["trained_at"] == before

    monkeypatch.setattr(app_module, "PROMOTION_TOLERANCE", 0.01)
    assert client.post("/retrain").json()["promoted"] is True


def test_actions_roundtrip(client):
    client.post("/actions", json={"customer_id": "c1", "action": "retention_call", "details": {"x": 1}})
    latest = client.get("/actions", params={"limit": 1}).json()["actions"][0]
    assert latest["customer_id"] == "c1" and latest["action"] == "retention_call"


def test_classify_unavailable_returns_503(client, monkeypatch):
    def unavailable(text):
        raise RuntimeError("complaints classifier unavailable (no torch)")
    monkeypatch.setattr(complaints, "classify", unavailable)
    assert client.post("/classify", json={"text": "late fee"}).status_code == 503


# --- units -------------------------------------------------------------------------------

def test_psi_zero_for_identical_distributions():
    X = pd.DataFrame({"a": list("xxyz") * 50, "n": np.arange(200.0)})
    profile = drift.build_profile(X, ["a"], ["n"])
    assert all(v < 1e-9 for v in drift.feature_psi(profile, X).values())


def test_complaint_text_normalized_like_training_data():
    assert complaints.normalize("My CARD was charged $35!!  Twice.") == "my card was charged twice"
