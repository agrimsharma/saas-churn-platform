"""
Publish the free public demo to Hugging Face (no card needed, ever):
  1. PUBLIC model repo  <user>/cfpb-complaints-distilbert  <- models/complaints-classifier
     (label names written into config.json, plus a model card)
  2. PUBLIC Docker Space <user>/churn-platform  <- deploy/hf-space + service/ + dashboard/ + reports
  3. Space settings: COMPLAINTS_MODEL variable; DATABASE_URL secret (Neon, optional)

Prereqs (you, once):  pip install huggingface_hub  &&  hf auth login   (a WRITE token)
Then:  NEON_DATABASE_URL='postgresql://...' python scripts/publish_hf.py
(copy the complaint index to Neon first: scripts/copy_index_to_postgres.sh)
"""
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.join(ROOT, "models", "complaints-classifier")
METRICS = os.path.join(ROOT, "models", "metrics.json")
SPACE_DIR = os.path.join(ROOT, "deploy", "hf-space")
REPORT_FILES = ["example_drafts.json", "demo_actions.jsonl", "retail_backtest.json",
                "rag_retrieval_eval.json", "drift_summary.json"]

MODEL_CARD = """---
license: apache-2.0
language: en
pipeline_tag: text-classification
base_model: distilbert-base-uncased
datasets: [CFPB/consumer-finance-complaints]
---
# CFPB complaint product classifier (DistilBERT)

Fine-tuned `distilbert-base-uncased` that sorts consumer financial complaints into 5 products:
{labels}. Trained on 162k CFPB complaint narratives (lowercased, punctuation stripped - normalize
input the same way), 3 epochs on an RTX 3060 Ti.

| | accuracy | macro-F1 |
|---|---|---|
| this model | {acc:.3f} | {f1:.3f} |
| TF-IDF + logistic regression | 0.846 | 0.818 |

Part of [saas-churn-platform](https://github.com/agrimsharma/saas-churn-platform).
"""


def main():
    api = HfApi()
    user = api.whoami()["name"]
    model_repo, space = f"{user}/cfpb-complaints-distilbert", f"{user}/churn-platform"
    metrics = json.load(open(METRICS))

    print(f"1/3 model {model_repo}")
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir(MODEL_DIR):
            if f != "training_args.bin":
                shutil.copy2(os.path.join(MODEL_DIR, f), tmp)
        cfg = json.load(open(os.path.join(tmp, "config.json")))
        cfg["id2label"] = {str(v): k for k, v in metrics["label_to_id"].items()}
        cfg["label2id"] = metrics["label_to_id"]
        json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"), indent=2)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(MODEL_CARD.format(labels=", ".join(metrics["label_to_id"]),
                                      acc=metrics["metrics"]["eval_accuracy"], f1=metrics["metrics"]["eval_macro_f1"]))
        api.upload_folder(repo_id=model_repo, folder_path=tmp, commit_message="Upload fine-tuned classifier")

    print(f"2/3 Space {space}")
    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir(SPACE_DIR):
            shutil.copy2(os.path.join(SPACE_DIR, f), tmp)
        for f in ("requirements-api.txt", "requirements-nlp.txt"):
            shutil.copy2(os.path.join(ROOT, f), tmp)
        ignore = shutil.ignore_patterns("__pycache__", "Dockerfile")
        shutil.copytree(os.path.join(ROOT, "service"), os.path.join(tmp, "service"), ignore=ignore)
        shutil.copytree(os.path.join(ROOT, "dashboard"), os.path.join(tmp, "dashboard"), ignore=ignore)
        os.makedirs(os.path.join(tmp, "reports"))
        for f in REPORT_FILES:
            shutil.copy2(os.path.join(ROOT, "reports", f), os.path.join(tmp, "reports", f))
        os.makedirs(os.path.join(tmp, "models"))
        shutil.copy2(METRICS, os.path.join(tmp, "models", "metrics.json"))
        api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp, commit_message="Deploy public demo")

    print("3/3 Space settings")
    api.add_space_variable(space, "COMPLAINTS_MODEL", model_repo)
    if os.environ.get("NEON_DATABASE_URL"):
        api.add_space_secret(space, "DATABASE_URL", os.environ["NEON_DATABASE_URL"])
    else:
        print("   ! no NEON_DATABASE_URL: similar-complaint search stays off until you add the "
              "DATABASE_URL secret in the Space settings")
    print(f"\nSpace: https://huggingface.co/spaces/{space}  (first build ~10 min)")
    print(f"App:   https://{user}-churn-platform.hf.space")


if __name__ == "__main__":
    main()
