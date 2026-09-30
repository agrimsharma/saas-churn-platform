import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from evidently import Report
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------------------------
# Drift + decay monitoring demo.
#
# We don't have real production data collected over time, so the "current"
# (post-deployment) window below is SIMULATED and clearly labeled as such,
# combining two realistic scenarios applied to the same held-out rows the
# XGBoost baseline was originally scored on:
#
#  1. A price increase: MonthlyCharges +25%, TotalCharges recomputed to stay
#     internally consistent.
#  2. A contract migration: a large share of existing One/Two-year customers
#     move to Month-to-month plans (e.g. a big promo, or a competitor's
#     offer) - this targets Contract type, the model's single most important
#     feature
#     (~58% combined importance for Contract_One/Two year), so it's the
#     scenario most likely to actually break its decision boundary rather
#     than just showing up as a cosmetic feature-distribution shift.
#
# In both cases labels are left untouched - we're testing whether a model
# trained on the old regime still holds up against inputs shifted the way a
# real deployment would drift, not asserting that churn behavior itself
# changed retroactively.
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn_with_all_feedback.csv")
MLFLOW_DB = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
REPORT_OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "drift_report.html")
XGB_RUN_ID = "c8bcc57139cf46f1b8f7ffcb89487c55"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}")

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
y = (df["Churn"] == "Yes").astype(int)

drop_cols = ["customerID", "Churn", "PromptInput", "CustomerFeedback"]
X_raw = df.drop(columns=drop_cols)
categorical_cols = X_raw.select_dtypes(include="object").columns.tolist()
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

# same split as train_baseline.py so "reference" here is exactly the test set
# the model's logged roc_auc (0.8486) was measured on
X_train_enc, X_test_enc, y_train, y_test, raw_train, raw_test = train_test_split(
    X_encoded, y, X_raw, test_size=0.2, random_state=42, stratify=y
)
print(f"Reference (original test set) size: {len(X_test_enc)}")

# --- build the simulated "current" (drifted) window -----------------------
raw_current = raw_test.copy()
raw_current["MonthlyCharges"] = raw_current["MonthlyCharges"] * 1.25
# TotalCharges tracks MonthlyCharges * tenure in the real data; keep that
# relationship consistent under the simulated price increase (tenure=0 rows
# stay at 0, same as the original data quirk)
raw_current["TotalCharges"] = np.where(
    raw_test["tenure"] > 0,
    raw_current["MonthlyCharges"] * raw_test["tenure"],
    0.0,
)

# Contract migration: 70% of existing One/Two-year customers move to
# Month-to-month, simulating a large promotional push (feature changes,
# true label - whether they actually churned - does not, since it hasn't
# "happened yet" in this snapshot)
rng = np.random.RandomState(42)
long_contract_idx = raw_current.index[raw_current["Contract"].isin(["One year", "Two year"])]
migrating_idx = rng.choice(long_contract_idx, size=int(len(long_contract_idx) * 0.7), replace=False)
raw_current.loc[migrating_idx, "Contract"] = "Month-to-month"

X_current_enc = pd.get_dummies(raw_current, columns=categorical_cols, drop_first=True)
# align columns in case a category's presence/absence shifts one-hot columns
X_current_enc = X_current_enc.reindex(columns=X_test_enc.columns, fill_value=0)

# --- load the trained XGBoost model and compare performance ---------------
model = mlflow.xgboost.load_model(f"runs:/{XGB_RUN_ID}/model")

ref_probs = model.predict_proba(X_test_enc)[:, 1]
ref_preds = model.predict(X_test_enc)
cur_probs = model.predict_proba(X_current_enc)[:, 1]
cur_preds = model.predict(X_current_enc)

ref_metrics = {
    "accuracy": accuracy_score(y_test, ref_preds),
    "roc_auc": roc_auc_score(y_test, ref_probs),
}
cur_metrics = {
    "accuracy": accuracy_score(y_test, cur_preds),
    "roc_auc": roc_auc_score(y_test, cur_probs),
}

print("\nModel performance:")
print(f"  Reference (original pricing): accuracy={ref_metrics['accuracy']:.4f}, roc_auc={ref_metrics['roc_auc']:.4f}")
print(f"  Current   (simulated +25% price hike): accuracy={cur_metrics['accuracy']:.4f}, roc_auc={cur_metrics['roc_auc']:.4f}")

roc_auc_drop = ref_metrics["roc_auc"] - cur_metrics["roc_auc"]
print(f"  ROC-AUC drop: {roc_auc_drop:.4f}")

# --- Evidently data drift report (human-readable raw columns, not dummies) -
os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)

report = Report([DataDriftPreset()])
snapshot = report.run(current_data=raw_current, reference_data=raw_test)
snapshot.save_html(REPORT_OUT)
print(f"\nEvidently drift report saved to {REPORT_OUT}")

drift_result = snapshot.dict()
drifted_count_metric = next(
    m for m in drift_result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")
)
drift_share_threshold = drifted_count_metric["config"]["drift_share"]
drifted_count = drifted_count_metric["value"]["count"]
drifted_share = drifted_count_metric["value"]["share"]
dataset_drift = drifted_share >= drift_share_threshold
print(f"Drift detected in {drifted_count:.0f} columns ({drifted_share:.1%}) "
      f"- dataset-level drift (share >= {drift_share_threshold:.0%}): {dataset_drift}")

per_column_drift = {}
for m in drift_result["metrics"]:
    if not m["metric_name"].startswith("ValueDrift"):
        continue
    col = m["config"]["column"]
    threshold = m["config"]["threshold"]
    value = m["value"]
    per_column_drift[col] = {"value": value, "threshold": threshold, "drifted": value >= threshold}

print("Per-column drift scores:")
for col, info in per_column_drift.items():
    print(f"  {col}: value={info['value']:.4f} threshold={info['threshold']} drifted={info['drifted']}")

# --- importance-weighted drift score ----------------------------------------
# The naive "share of columns drifted" metric above treats every column
# equally - it flagged only 15.8% of columns as drifted (below its own 50%
# dataset-drift threshold) even though one of those columns is Contract, the
# model's single most important feature (~58% of total importance). That
# naive share metric would badly understate risk here. Instead, weight each
# drifted raw column by how much the model actually relies on it.
importances = dict(zip(model.get_booster().feature_names, model.feature_importances_))


def raw_column_importance(raw_col):
    if raw_col in categorical_cols:
        return sum(v for k, v in importances.items() if k.startswith(f"{raw_col}_"))
    return importances.get(raw_col, 0.0)


drifted_importance_mass = sum(
    raw_column_importance(col) for col, info in per_column_drift.items() if info["drifted"]
)
print(f"\nImportance-weighted drift score (share of model's total feature "
      f"importance resting on drifted columns): {drifted_importance_mass:.1%}")

# --- retrain-trigger logic --------------------------------------------------
# Fire on EITHER signal: a measurable drop in the model's own held-out
# performance, OR drift concentrated in features the model heavily relies on
# (a leading indicator - catching risk before it fully shows up in aggregate
# metrics like ROC-AUC, which can be fairly insensitive to a shift confined
# to one feature when the model's other features still rank correctly).
ROC_AUC_DROP_THRESHOLD = 0.03
IMPORTANCE_DRIFT_THRESHOLD = 0.3

performance_trigger = roc_auc_drop > ROC_AUC_DROP_THRESHOLD
importance_drift_trigger = drifted_importance_mass > IMPORTANCE_DRIFT_THRESHOLD
retrain_needed = performance_trigger or importance_drift_trigger

print(f"\nRetrain trigger:")
print(f"  performance drop (roc_auc drop > {ROC_AUC_DROP_THRESHOLD}): {performance_trigger}")
print(f"  importance-weighted drift (> {IMPORTANCE_DRIFT_THRESHOLD:.0%}): {importance_drift_trigger}")
print(f"  => {'RETRAIN RECOMMENDED' if retrain_needed else 'model healthy'}")

summary = {
    "reference_metrics": ref_metrics,
    "current_metrics": cur_metrics,
    "roc_auc_drop": roc_auc_drop,
    "roc_auc_drop_threshold": ROC_AUC_DROP_THRESHOLD,
    "performance_trigger": bool(performance_trigger),
    "drifted_columns_count": drifted_count,
    "drifted_columns_share": drifted_share,
    "dataset_level_drift_naive": bool(dataset_drift),
    "drifted_importance_mass": float(drifted_importance_mass),
    "importance_drift_threshold": IMPORTANCE_DRIFT_THRESHOLD,
    "importance_drift_trigger": bool(importance_drift_trigger),
    "retrain_recommended": bool(retrain_needed),
    "simulated_scenario": (
        "MonthlyCharges +25% (price increase, TotalCharges recomputed consistently) + "
        "70% of One/Two-year contract customers migrated to Month-to-month (large promotional-push "
        "scenario, targeting the model's top feature); labels unchanged in both cases"
    ),
}
SUMMARY_OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "drift_summary.json")
with open(SUMMARY_OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {SUMMARY_OUT}")
