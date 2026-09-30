"""Churn scoring model: training, loading, and per-customer risk explanations.

Unlike scripts/train_baseline.py (which one-hot encodes with pd.get_dummies and
logs a bare XGBoost model), this saves preprocessing + model as ONE sklearn
Pipeline, so the API can score raw customer records exactly as they arrive
from a CRM - no column-alignment step that can silently drift out of sync.

The saved bundle also holds:
  * an isotonic calibrator fit on out-of-fold predictions - the class-weighted
    XGBoost ranks well but its probabilities run high, and the retention
    playbook needs "0.6" to actually mean ~60%
  * the training-data profile and per-feature importances used by drift.py
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBClassifier

from service import drift

ROOT = Path(__file__).resolve().parents[1]
# any IBM Telco churn CSV works (e.g. the public Telco-Customer-Churn.csv); the extra
# LLM-feedback columns in the default file are ignored
DATA_PATH = Path(os.environ.get("CHURN_DATA_PATH", ROOT / "data" / "raw" / "telco_churn_with_all_feedback.csv"))
MODEL_DIR = ROOT / "models" / "churn_scoring"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
META_PATH = MODEL_DIR / "meta.json"

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = CATEGORICAL + NUMERIC

# same hyperparameters as the logged XGBoost baseline, so results stay comparable
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = df[FEATURES].copy()
    # known Telco quirk: TotalCharges is blank for tenure=0 customers (no bill yet)
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    return X


def load_training_data():
    df = pd.read_csv(DATA_PATH)
    return clean(df), (df["Churn"] == "Yes").astype(int)


def build_pipeline(y: pd.Series) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",  # NUMERIC, in FEATURES order after the categoricals
    )
    clf = XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), **XGB_PARAMS)
    return Pipeline([("pre", pre), ("clf", clf)])


def _oof_probs(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Out-of-fold raw probabilities + per-fold ROC-AUC."""
    oof = np.zeros(len(y))
    aucs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y):
        p = build_pipeline(y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y.iloc[te], p))
    return oof, np.array(aucs)


def _fit_calibrator(raw: np.ndarray, y: pd.Series) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)


def feature_importances(pipeline: Pipeline) -> Dict[str, float]:
    """Model importance per original CRM field (one-hot columns summed back)."""
    imp = pipeline.named_steps["clf"].feature_importances_
    per_col = pd.Series(imp).groupby(np.array(_raw_column_per_encoded_feature(pipeline))).sum()
    return (per_col / per_col.sum()).round(6).to_dict()


def train() -> Dict:
    """Evaluate on a held-out split + 5-fold CV, then refit on all rows for serving."""
    X, y = load_training_data()

    # held-out evaluation, including whether calibration actually helps on unseen rows
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    holdout = build_pipeline(y_tr).fit(X_tr, y_tr)
    preds = holdout.predict(X_te)
    raw_te = holdout.predict_proba(X_te)[:, 1]
    cal_te = _fit_calibrator(_oof_probs(X_tr, y_tr)[0], y_tr).predict(raw_te)

    oof, cv_auc = _oof_probs(X, y)
    pipeline = build_pipeline(y).fit(X, y)
    model = {
        "pipeline": pipeline,
        "calibrator": _fit_calibrator(oof, y),
        "profile": drift.build_profile(X, CATEGORICAL, NUMERIC),
        "importances": feature_importances(pipeline),
    }
    model["meta"] = {
        "sklearn_version": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_path": DATA_PATH.name,
        "n_rows": int(len(X)),
        "churn_rate": float(y.mean()),
        "holdout_metrics": {
            # decision metrics use the raw class-weighted model's 0.5 cut (recall-oriented)
            "accuracy": accuracy_score(y_te, preds),
            "precision": precision_score(y_te, preds),
            "recall": recall_score(y_te, preds),
            "f1": f1_score(y_te, preds),
            "roc_auc": roc_auc_score(y_te, raw_te),
            "brier_raw": brier_score_loss(y_te, raw_te),
            "brier_calibrated": brier_score_loss(y_te, cal_te),
            "mean_predicted_raw": float(raw_te.mean()),
            "mean_predicted_calibrated": float(cal_te.mean()),
            "observed_churn_rate": float(y_te.mean()),
        },
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "params": XGB_PARAMS,
        "features": FEATURES,
        "top_features": dict(sorted(model["importances"].items(), key=lambda kv: -kv[1])[:5]),
    }
    return model


def save(model: Dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # write-then-rename so a running API never reads a half-written file
    tmp = PIPELINE_PATH.with_suffix(".tmp")
    joblib.dump({k: v for k, v in model.items() if k != "meta"}, tmp)
    tmp.replace(PIPELINE_PATH)
    META_PATH.write_text(json.dumps(model["meta"], indent=2))


def load() -> Optional[Dict]:
    if not PIPELINE_PATH.exists():
        return None
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    # pickled sklearn pipelines aren't portable across sklearn versions (e.g. a
    # model trained on the host, then mounted into the Docker image) - retrain instead
    if meta.get("sklearn_version") != sklearn.__version__:
        return None
    bundle = joblib.load(PIPELINE_PATH)
    if not isinstance(bundle, dict) or "calibrator" not in bundle:
        return None  # pre-calibration format
    return {**bundle, "meta": meta}


def _raw_column_per_encoded_feature(pipeline: Pipeline) -> List[str]:
    enc = pipeline.named_steps["pre"].named_transformers_["cat"]
    cols = []
    for col, cats in zip(CATEGORICAL, enc.categories_):
        cols.extend([col] * len(cats))
    return cols + NUMERIC


def score(model: Dict, records: List[Dict], top_n: int = 3) -> List[Dict]:
    """Calibrated churn probability plus the features pushing each customer's risk up.

    Uses XGBoost's exact per-prediction SHAP contributions (pred_contribs),
    summed back from one-hot columns to the original CRM fields, so a
    retention rep sees "Contract = Month-to-month" rather than an encoded
    column name.
    """
    pipeline = model["pipeline"]
    df = pd.DataFrame(records)
    X = clean(df)
    probs = model["calibrator"].predict(pipeline.predict_proba(X)[:, 1])

    Xt = pipeline.named_steps["pre"].transform(X)
    contribs = pipeline.named_steps["clf"].get_booster().predict(DMatrix(Xt), pred_contribs=True)[:, :-1]  # drop bias
    raw_cols = np.array(_raw_column_per_encoded_feature(pipeline))

    results = []
    for i, prob in enumerate(probs):
        per_col = pd.Series(contribs[i]).groupby(raw_cols).sum().sort_values(ascending=False)
        factors = [
            {"feature": col, "value": _jsonable(X.iloc[i][col]), "impact": round(float(v), 4)}
            for col, v in per_col.head(top_n).items() if v > 0
        ]
        results.append({
            "customer_id": df["customerID"].iloc[i] if "customerID" in df else None,
            "churn_probability": round(float(prob), 4),
            "top_risk_factors": factors,
        })
    return results


def _jsonable(v):
    return v.item() if isinstance(v, np.generic) else v
