"""Churn platform API - the HTTP surface n8n workflows call.

Run locally:  uvicorn service.app:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from service import churn_model, complaints, drift

ROOT = Path(__file__).resolve().parents[1]
DRIFT_SUMMARY_PATH = ROOT / "reports" / "drift_summary.json"
ACTIONS_LOG_PATH = ROOT / "reports" / "actions.jsonl"
API_KEY = os.environ.get("API_KEY")  # unset = no auth (local dev)
# a retrained model is only promoted if its CV ROC-AUC is at most this much worse
PROMOTION_TOLERANCE = float(os.environ.get("PROMOTION_TOLERANCE", "0.01"))

_state: Dict[str, Any] = {"model": None}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = churn_model.load()
    if loaded is None:
        print("No compatible churn model found - training one (a few seconds)...")
        loaded = churn_model.train()
        churn_model.save(loaded)
    _state["model"] = loaded
    yield


app = FastAPI(title="SaaS Churn Platform API", version="0.2.0", lifespan=lifespan)


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class Customer(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None


class ScoreRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=5000)


class DriftCheckRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=drift.MIN_BATCH, max_length=50000)
    # optional outcomes for these customers (e.g. from a later billing export) - enables the
    # performance check on top of the label-free feature drift check
    churned: Optional[List[bool]] = None


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class Action(BaseModel):
    customer_id: Optional[str] = None
    action: str
    details: Dict[str, Any] = {}
    source: str = "n8n"


@app.get("/health")
def health():
    meta = _state["model"]["meta"] if _state["model"] else {}
    return {
        "status": "ok",
        "churn_model": {
            "trained_at": meta.get("trained_at"),
            "cv_roc_auc_mean": meta.get("cv_roc_auc_mean"),
        },
        # don't force the (slow) DistilBERT load just for a health check
        "complaints_classifier_loaded": complaints._model is not None,
    }


@app.post("/score", dependencies=[Depends(require_key)])
def score(req: ScoreRequest):
    records = [c.model_dump() for c in req.customers]
    with _lock:
        model = _state["model"]
    return {
        "model_trained_at": model["meta"].get("trained_at"),
        "predictions": churn_model.score(model, records),
    }


@app.post("/classify", dependencies=[Depends(require_key)])
async def classify(req: ClassifyRequest):
    try:
        return await run_in_threadpool(complaints.classify, req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/drift/check", dependencies=[Depends(require_key)])
def drift_check(req: DriftCheckRequest):
    """Live drift check of a customer batch against the model's training data."""
    if req.churned is not None and len(req.churned) != len(req.customers):
        raise HTTPException(status_code=422, detail="churned must have one entry per customer")
    with _lock:
        model = _state["model"]
    X = churn_model.clean(pd.DataFrame([c.model_dump() for c in req.customers]))
    probs = model["pipeline"].predict_proba(X)[:, 1] if req.churned is not None else None
    result = drift.check(model["profile"], model["importances"], X, probs, req.churned,
                         model["meta"].get("cv_roc_auc_mean"))
    result["model_trained_at"] = model["meta"].get("trained_at")
    return result


@app.get("/drift", dependencies=[Depends(require_key)])
def drift_report():
    """The original one-off Evidently report (simulated scenario, scripts/drift_monitoring.py)."""
    if not DRIFT_SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="no drift summary - run scripts/drift_monitoring.py")
    summary = json.loads(DRIFT_SUMMARY_PATH.read_text())
    summary["generated_at"] = datetime.fromtimestamp(
        DRIFT_SUMMARY_PATH.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    return summary


@app.post("/retrain", dependencies=[Depends(require_key)])
async def retrain():
    """Retrain on the current data file; promote only if CV ROC-AUC holds up."""
    # NOTE: retrains on CHURN_DATA_PATH. In production that file would be refreshed with
    # newly labeled customers before this is called; here it's the static Telco export.
    candidate = await run_in_threadpool(churn_model.train)
    old_auc = _state["model"]["meta"].get("cv_roc_auc_mean", 0.0)
    new_auc = candidate["meta"]["cv_roc_auc_mean"]
    promoted = new_auc >= old_auc - PROMOTION_TOLERANCE
    if promoted:
        churn_model.save(candidate)
        with _lock:
            _state["model"] = candidate
    return {
        "promoted": promoted,
        "previous_cv_roc_auc": old_auc,
        "candidate_cv_roc_auc": new_auc,
        "candidate_holdout_metrics": candidate["meta"]["holdout_metrics"],
    }


@app.get("/demo/customers", dependencies=[Depends(require_key)])
def demo_customers(
    n: int = Query(25, ge=1, le=5000),
    seed: Optional[int] = None,
    scenario: Literal["none", "price_hike", "contract_shift", "both"] = "none",
    include_labels: bool = False,
):
    """Stand-in for a CRM "active customers" export: random Telco rows, labels removed.

    scenario applies the simulated drift from scripts/drift_monitoring.py (+25% prices,
    70% of 1/2-year contracts moved to month-to-month) so the drift workflow has
    something to detect. include_labels returns true outcomes as `churned`, for the
    drift check's performance test.
    """
    df = pd.read_csv(churn_model.DATA_PATH)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample["TotalCharges"] = pd.to_numeric(sample["TotalCharges"], errors="coerce").fillna(0)
    if scenario != "none":
        sample = drift.simulate(sample, scenario, seed=seed or 42)
    out = {"scenario": scenario,
           "customers": sample[["customerID"] + churn_model.FEATURES].to_dict(orient="records")}
    if include_labels:
        out["churned"] = (sample["Churn"] == "Yes").tolist()
    return out


@app.post("/actions", dependencies=[Depends(require_key)])
def log_action(action: Action):
    """Local action log - where a real deployment would write to the CRM."""
    entry = {"logged_at": datetime.now(timezone.utc).isoformat(), **action.model_dump()}
    ACTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, ACTIONS_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


@app.get("/actions", dependencies=[Depends(require_key)])
def list_actions(limit: int = Query(50, ge=1, le=1000)):
    if not ACTIONS_LOG_PATH.exists():
        return {"actions": []}
    lines = ACTIONS_LOG_PATH.read_text().splitlines()[-limit:]
    return {"actions": [json.loads(line) for line in reversed(lines)]}
