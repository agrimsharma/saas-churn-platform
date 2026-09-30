import os
import json
import mlflow

MLFLOW_DB = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "complaints-classifier")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}")
mlflow.set_experiment("complaints-classifier-finetune")

with open(METRICS_PATH) as f:
    result = json.load(f)

with mlflow.start_run(run_name="distilbert_complaints_full_gpu"):
    mlflow.log_params({
        "model": "distilbert-base-uncased",
        "dataset": "CFPB consumer complaints (real, 162421 rows)",
        "epochs": 3,
        "batch_size": 32,
        "trained_on": "vast.ai RTX 3060 Ti (GPU)",
        "label_to_id": json.dumps(result["label_to_id"]),
        "note": "Standalone NLP/fine-tuning technique demo - real text, not fused with the "
                "Telco churn tabular model (see docs note on why: synthetic-text leakage found "
                "in the original churn-feedback dataset)",
    })
    mlflow.log_metrics({
        "accuracy": result["metrics"]["eval_accuracy"],
        "macro_f1": result["metrics"]["eval_macro_f1"],
        "macro_precision": result["metrics"]["eval_macro_precision"],
        "macro_recall": result["metrics"]["eval_macro_recall"],
        "eval_loss": result["metrics"]["eval_loss"],
    })
    mlflow.log_artifacts(MODEL_DIR, artifact_path="model")

print("Logged to MLflow experiment 'complaints-classifier-finetune'")
print(f"Final: accuracy={result['metrics']['eval_accuracy']:.4f}, macro_f1={result['metrics']['eval_macro_f1']:.4f}")
print("Baseline (TF-IDF+LogReg) was: accuracy=0.846, macro_f1=0.818")
