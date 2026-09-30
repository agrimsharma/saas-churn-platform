import os
import numpy as np
import pandas as pd
import mlflow
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
)
from datasets import Dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "telco_text_masked.csv")
MLFLOW_DB = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert-churn-text")
MODEL_NAME = "distilbert-base-uncased"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}")
mlflow.set_experiment("churn-text-finetune")

df = pd.read_csv(DATA_PATH)
y = (df["Churn"] == "Yes").astype(int)
texts = df["CustomerFeedback_masked"].tolist()

train_texts, test_texts, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


train_ds = Dataset.from_dict({"text": train_texts, "label": y_train.tolist()}).map(tokenize, batched=True)
test_ds = Dataset.from_dict({"text": test_texts, "label": y_test.tolist()}).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# class imbalance (~73/27) - weight the loss so the model doesn't just predict "no churn" always
class_counts = np.bincount(y_train)
class_weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float)
print(f"Class weights: {class_weights.tolist()}")


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "..", "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to=[],
)

trainer = WeightedTrainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

with mlflow.start_run(run_name="distilbert_text_only"):
    mlflow.log_params({
        "model": MODEL_NAME, "epochs": 3, "batch_size": 16, "max_length": 128,
        "note": "text-only (masked feedback), no tabular features",
    })
    trainer.train()
    metrics = trainer.evaluate()
    clean_metrics = {k.replace("eval_", ""): v for k, v in metrics.items()
                      if k.replace("eval_", "") in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    mlflow.log_metrics(clean_metrics)

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    trainer.save_model(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)

    print("\nText-only DistilBERT results:")
    for k, v in clean_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nModel saved to {MODEL_OUT_DIR}")
    print("Compare against tabular baseline: XGBoost ROC-AUC was 0.8486")
