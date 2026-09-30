"""
Publish the fine-tuned complaints classifier to the Hugging Face Hub (free, public):
  <user>/cfpb-complaints-distilbert  <- models/complaints-classifier
with the label names written into config.json and a model card. The Modal demo
(deploy/modal_app.py) and the Kubernetes chart load it from there.

Prereq (once):  pip install huggingface_hub  &&  hf auth login   (a WRITE token)
"""
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.join(ROOT, "models", "complaints-classifier")
METRICS = os.path.join(ROOT, "models", "metrics.json")

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
    model_repo = f"{user}/cfpb-complaints-distilbert"
    metrics = json.load(open(METRICS))

    print(f"Uploading {model_repo}")
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
    print(f"https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
