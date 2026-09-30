"""
Month-by-month backtest of the platform's monitoring + retraining loop on REAL time-stamped data.

Telco is a single snapshot, so drift there could only be simulated. UCI Online Retail II has
two years of transactions (Dec 2009 - Dec 2011, 5.9k customers), so we can replay what the
platform would actually have done each month.

Setup
  * Snapshot at each month start T: every customer who bought in the previous 180 days, with
    features computed ONLY from transactions before T (recency, frequency, spend, 90-day trends,
    returns, basket size). Label: no purchase in [T, T+90 days) = churned.
  * A model deployed at T may only train on snapshots whose labels had matured by T
    (cutoff <= T - 90 days) - no peeking at outcomes that haven't happened yet.
  * Each month the deployed model scores the population. Its TRUE ROC-AUC at T is what we
    report (known in hindsight). The monitoring signals it can actually see at T are:
      - PSI drift of this month's features vs its training data, importance-weighted (drift.py)
      - ROC-AUC on the newest matured snapshot (T - 90 days) vs its validation AUC
Policies compared
  static            train once, never retrain
  monthly           retrain every month on all matured snapshots
  drift_triggered   retrain only when the platform's rule fires (drift mass > 0.3 or
                    matured AUC drop > 0.03) - the loop n8n workflow 03 runs

Usage: python scripts/retail_backtest.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from service import drift  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "processed", "online_retail_ii.parquet")
REPORT = os.path.join(ROOT, "reports", "retail_backtest.json")
HORIZON = pd.Timedelta(days=90)
ACTIVE_WINDOW = pd.Timedelta(days=180)
# every feature uses the same FIXED 180-day look-back, and the first snapshot is the first month
# with a full window of history (data starts Dec 2009). Lifetime totals, uncapped tenure, or a
# window that isn't full yet all grow month after month purely because of the dataset's start
# date - "drift" that is an artifact, not a change in customer behaviour.
FEATURES = ["recency_days", "tenure_days_capped", "invoices_180d", "invoices_90d", "invoices_prev_90d",
            "spend_180d", "spend_90d", "spend_prev_90d", "avg_invoice_value", "distinct_products_180d",
            "return_rate", "is_uk"]
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
TOP_FRACTION = 0.10  # retention team calls the top 10% riskiest customers


def load_transactions() -> pd.DataFrame:
    df = pd.read_parquet(DATA).dropna(subset=["Customer ID"])
    df = df[df["Price"] > 0]
    df["customer"] = df["Customer ID"].astype(int)
    df["cancel"] = df["Invoice"].str.startswith("C")
    df["value"] = df["Quantity"] * df["Price"]
    return df[["customer", "Invoice", "InvoiceDate", "StockCode", "value", "cancel", "Country"]]


def snapshot(tx: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    past = tx[(tx["InvoiceDate"] < T) & (tx["InvoiceDate"] >= T - ACTIVE_WINDOW)]
    buys = past[~past["cancel"]]
    inv = buys.groupby(["customer", "Invoice"]).agg(date=("InvoiceDate", "min"), value=("value", "sum")).reset_index()
    g = inv.groupby("customer")
    f = pd.DataFrame({
        "last": g["date"].max(), "first": g["date"].min(),
        "invoices_180d": g.size(), "spend_180d": g["value"].sum(),
    })
    recent = inv[inv["date"] >= T - HORIZON].groupby("customer")
    prev = inv[(inv["date"] >= T - 2 * HORIZON) & (inv["date"] < T - HORIZON)].groupby("customer")
    f["invoices_90d"] = recent.size().reindex(f.index, fill_value=0)
    f["spend_90d"] = recent["value"].sum().reindex(f.index, fill_value=0.0)
    f["invoices_prev_90d"] = prev.size().reindex(f.index, fill_value=0)
    f["spend_prev_90d"] = prev["value"].sum().reindex(f.index, fill_value=0.0)
    f["recency_days"] = (T - f["last"]).dt.days
    f["tenure_days_capped"] = (T - f["first"]).dt.days  # first purchase within the window
    f["avg_invoice_value"] = f["spend_180d"] / f["invoices_180d"]
    f["distinct_products_180d"] = buys.groupby("customer")["StockCode"].nunique().reindex(f.index, fill_value=0)
    cancels = past[past["cancel"]].groupby("customer")["Invoice"].nunique().reindex(f.index, fill_value=0)
    f["return_rate"] = cancels / (f["invoices_180d"] + cancels)
    f["is_uk"] = (past.groupby("customer")["Country"].agg(lambda s: s.mode().iat[0]) == "United Kingdom"
                  ).reindex(f.index).astype(int)

    future = tx[(tx["InvoiceDate"] >= T) & (tx["InvoiceDate"] < T + HORIZON) & ~tx["cancel"]]
    f["churned"] = (~f.index.isin(future["customer"].unique())).astype(int)
    f["cutoff"] = T
    return f[FEATURES + ["churned", "cutoff"]].reset_index()


def fit(train: pd.DataFrame) -> XGBClassifier:
    y = train["churned"]
    return XGBClassifier(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1), **XGB_PARAMS).fit(train[FEATURES], y)


class Deployed:
    """A model plus everything the monitor compares against."""

    def __init__(self, snaps, T):
        matured = [s for s in snaps if s["cutoff"].iat[0] <= T - HORIZON]
        # validation AUC: train on all but the newest matured snapshot, score that one
        self.val_auc = roc_auc_score(matured[-1]["churned"], fit(pd.concat(matured[:-1])).predict_proba(
            matured[-1][FEATURES])[:, 1]) if len(matured) > 1 else None
        train = pd.concat(matured)
        self.model = fit(train)
        self.trained_at = T
        self.n_train = len(train)
        self.profile = drift.build_profile(train, [], FEATURES)
        imp = pd.Series(self.model.feature_importances_, index=FEATURES)
        self.importances = (imp / imp.sum()).to_dict()

    def score(self, X):
        return self.model.predict_proba(X[FEATURES])[:, 1]


def top_precision(y, p, frac=TOP_FRACTION):
    n = max(int(len(p) * frac), 1)
    return float(np.asarray(y)[np.argsort(-p)[:n]].mean())


def main():
    tx = load_transactions()
    cutoffs = pd.date_range("2010-06-01", "2011-09-01", freq="MS")  # full 180-day window from June 2010
    snaps = [snapshot(tx, T) for T in cutoffs]
    by_T = dict(zip(cutoffs, snaps))
    print("month      customers  churn_rate")
    for T, s in by_T.items():
        print(f"{T:%Y-%m}  {len(s):9,}  {s['churned'].mean():.3f}")

    # first deployment needs >= 2 matured snapshots so it has a validation AUC to monitor against
    deploy_months = [T for T in cutoffs if T >= pd.Timestamp("2010-11-01")]
    policies = {name: Deployed(snaps, deploy_months[0]) for name in ("static", "monthly", "drift_triggered")}
    rows, retrains = [], {k: [] for k in policies}

    for T in deploy_months:
        X = by_T[T]
        # newest snapshot whose 90-day outcome is known at T
        matured = [c for c in cutoffs if c <= T - HORIZON]
        matured_T = matured[-1] if matured else None
        for name, dep in list(policies.items()):
            if name == "monthly" and T != deploy_months[0]:
                dep = policies[name] = Deployed(snaps, T)
                retrains[name].append(f"{T:%Y-%m}")

            psi = drift.feature_psi(dep.profile, X[FEATURES])
            drifted = [c for c, v in psi.items() if v > drift.PSI_THRESHOLD]
            mass = sum(dep.importances.get(c, 0) for c in drifted)
            matured_auc = None
            # ...and that wasn't part of this model's training data
            if matured_T is not None and matured_T > dep.trained_at - HORIZON:
                m = by_T[matured_T]
                matured_auc = roc_auc_score(m["churned"], dep.score(m))
            auc_drop = (dep.val_auc - matured_auc) if (matured_auc is not None and dep.val_auc) else None
            fire = mass > drift.IMPORTANCE_DRIFT_THRESHOLD or (auc_drop is not None and auc_drop > drift.ROC_AUC_DROP_THRESHOLD)

            if name == "drift_triggered" and fire and T != dep.trained_at:
                dep = policies[name] = Deployed(snaps, T)
                retrains[name].append(f"{T:%Y-%m}")

            p = dep.score(X)
            rows.append({
                "month": f"{T:%Y-%m}", "policy": name, "customers": int(len(X)),
                "churn_rate": round(float(X["churned"].mean()), 4),
                "true_roc_auc": round(float(roc_auc_score(X["churned"], p)), 4),
                "top10_precision": round(top_precision(X["churned"], p), 4),
                "model_trained": f"{dep.trained_at:%Y-%m}",
                "drifted_features": drifted, "drift_mass": round(float(mass), 3),
                "matured_auc": round(float(matured_auc), 4) if matured_auc is not None else None,
                "trigger_fired": bool(fire),
            })

    res = pd.DataFrame(rows)
    summary = {}
    for name in policies:
        r = res[res.policy == name]
        summary[name] = {
            "mean_true_roc_auc": round(float(r.true_roc_auc.mean()), 4),
            "min_true_roc_auc": round(float(r.true_roc_auc.min()), 4),
            "mean_top10_precision": round(float(r.top10_precision.mean()), 4),
            "retrains": len(retrains[name]), "retrain_months": retrains[name],
        }
    pivot = res.pivot(index="month", columns="policy", values="true_roc_auc")
    dt = res[res.policy == "drift_triggered"].set_index("month")
    pivot["drift_fired"] = dt["trigger_fired"]
    pivot["drifted_features"] = dt["drifted_features"].map(lambda l: ",".join(l))
    print("\nTrue ROC-AUC by month (drift columns = drift_triggered policy's monitor):")
    print(pivot.to_string())
    print("\n" + json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump({"horizon_days": 90, "deploy_from": f"{deploy_months[0]:%Y-%m}", "features": FEATURES,
                   "summary": summary, "monthly": rows}, f, indent=2)
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
