"""Live, label-free drift check: current customers vs. the data the model was trained on.

Replaces the one-off Evidently report (scripts/drift_monitoring.py) for the serving path:
it runs on any batch the caller sends (n8n pulls it from the CRM), needs no extra
dependency, and keeps the importance-weighted trigger from that script - drift in a
feature the model barely uses shouldn't trigger a retrain, drift in Contract should.

Per-feature drift is the Population Stability Index (PSI): < 0.1 stable, 0.1-0.25
moderate, > 0.25 major. A feature counts as drifted above PSI_THRESHOLD.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_THRESHOLD = 0.2
IMPORTANCE_DRIFT_THRESHOLD = 0.3   # share of model importance resting on drifted features
ROC_AUC_DROP_THRESHOLD = 0.03      # only checked when labels are supplied
MIN_BATCH = 100                    # PSI on tiny batches is mostly noise
_EPS = 1e-4


def build_profile(X: pd.DataFrame, categorical: List[str], numeric: List[str]) -> Dict:
    """Reference distributions, stored with the model at training time."""
    profile = {}
    for col in categorical:
        profile[col] = {"type": "categorical",
                        "freq": X[col].astype(str).value_counts(normalize=True).to_dict()}
    for col in numeric:
        edges = np.unique(np.quantile(X[col].astype(float), np.linspace(0, 1, 11)))
        counts = np.histogram(X[col].astype(float), bins=_open_edges(edges))[0]
        profile[col] = {"type": "numeric", "edges": edges.tolist(),
                        "freq": (counts / counts.sum()).tolist()}
    return profile


def _open_edges(edges):
    # extend the outer bins to +/-inf so out-of-range current values still land in a bin
    e = np.asarray(edges, dtype=float).copy()
    e[0], e[-1] = -np.inf, np.inf
    return e


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(ref, _EPS, None)
    cur = np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_psi(profile: Dict, X: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for col, p in profile.items():
        if p["type"] == "categorical":
            cats = sorted(set(p["freq"]) | set(X[col].astype(str).unique()))
            cur = X[col].astype(str).value_counts(normalize=True)
            out[col] = psi(np.array([p["freq"].get(c, 0.0) for c in cats]),
                           np.array([cur.get(c, 0.0) for c in cats]))
        else:
            counts = np.histogram(X[col].astype(float), bins=_open_edges(p["edges"]))[0]
            out[col] = psi(np.array(p["freq"]), counts / max(counts.sum(), 1))
    return out


def check(profile: Dict, importances: Dict[str, float], X: pd.DataFrame,
          probs: Optional[np.ndarray] = None, labels: Optional[List[bool]] = None,
          reference_roc_auc: Optional[float] = None) -> Dict:
    if len(X) < MIN_BATCH:
        raise ValueError(f"need at least {MIN_BATCH} customers for a stable drift estimate, got {len(X)}")

    scores = feature_psi(profile, X)
    drifted = sorted((c for c, s in scores.items() if s > PSI_THRESHOLD), key=lambda c: -scores[c])
    mass = float(sum(importances.get(c, 0.0) for c in drifted))

    roc_auc = roc_auc_drop = None
    if labels is not None and probs is not None and 0 < sum(labels) < len(labels):
        roc_auc = float(roc_auc_score(np.asarray(labels, dtype=int), probs))
        roc_auc_drop = float(reference_roc_auc - roc_auc) if reference_roc_auc is not None else None

    importance_trigger = mass > IMPORTANCE_DRIFT_THRESHOLD
    performance_trigger = roc_auc_drop is not None and roc_auc_drop > ROC_AUC_DROP_THRESHOLD
    return {
        "n_customers": int(len(X)),
        "psi_threshold": PSI_THRESHOLD,
        "feature_psi": {c: round(s, 4) for c, s in sorted(scores.items(), key=lambda kv: -kv[1])},
        "drifted_features": drifted,
        "drifted_importance_mass": round(mass, 4),
        "importance_drift_threshold": IMPORTANCE_DRIFT_THRESHOLD,
        "importance_drift_trigger": importance_trigger,
        "current_roc_auc": roc_auc,
        "roc_auc_drop": roc_auc_drop,
        "roc_auc_drop_threshold": ROC_AUC_DROP_THRESHOLD,
        "performance_trigger": performance_trigger,
        "retrain_recommended": bool(importance_trigger or performance_trigger),
    }


def simulate(df: pd.DataFrame, scenario: str, seed: int = 42) -> pd.DataFrame:
    """Demo drift scenarios - the same shifts scripts/drift_monitoring.py simulated."""
    out = df.copy()
    rng = np.random.RandomState(seed)
    if scenario in ("price_hike", "both"):
        out["MonthlyCharges"] = out["MonthlyCharges"] * 1.25
        out["TotalCharges"] = np.where(out["tenure"] > 0, out["MonthlyCharges"] * out["tenure"], 0.0)
    if scenario in ("contract_shift", "both"):
        long_idx = out.index[out["Contract"].isin(["One year", "Two year"])]
        move = rng.choice(long_idx, size=int(len(long_idx) * 0.7), replace=False)
        out.loc[move, "Contract"] = "Month-to-month"
    return out
