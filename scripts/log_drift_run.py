import os
import json
import mlflow

MLFLOW_DB = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "drift_summary.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "drift_report.html")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}")
mlflow.set_experiment("drift-decay-monitoring")

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

with mlflow.start_run(run_name="simulated_price_hike_plus_contract_migration"):
    mlflow.log_params({
        "base_model": "xgboost (run c8bcc57139cf46f1b8f7ffcb89487c55, churn-baseline-tabular)",
        "simulated_scenario": summary["simulated_scenario"],
        "roc_auc_drop_threshold": summary["roc_auc_drop_threshold"],
        "importance_drift_threshold": summary["importance_drift_threshold"],
    })
    mlflow.log_metrics({
        "reference_accuracy": summary["reference_metrics"]["accuracy"],
        "reference_roc_auc": summary["reference_metrics"]["roc_auc"],
        "current_accuracy": summary["current_metrics"]["accuracy"],
        "current_roc_auc": summary["current_metrics"]["roc_auc"],
        "roc_auc_drop": summary["roc_auc_drop"],
        "drifted_columns_share": summary["drifted_columns_share"],
        "drifted_importance_mass": summary["drifted_importance_mass"],
        "performance_trigger": int(summary["performance_trigger"]),
        "importance_drift_trigger": int(summary["importance_drift_trigger"]),
        "retrain_recommended": int(summary["retrain_recommended"]),
    })
    mlflow.log_artifact(REPORT_PATH, artifact_path="drift_report")
    mlflow.log_artifact(SUMMARY_PATH, artifact_path="drift_report")

print("Logged to MLflow experiment 'drift-decay-monitoring'")
