"""Complaint category classifier (fine-tuned DistilBERT, models/complaints-classifier).

Loaded lazily and optional: torch + transformers are ~1 GB, so the API still
serves /score without them (build the image with WITH_NLP=true to enable).
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "complaints-classifier"
# a local dir, or a Hugging Face Hub repo id (the free public demo loads it from the Hub)
MODEL_SOURCE = os.environ.get("COMPLAINTS_MODEL", str(MODEL_DIR))
METRICS_PATH = ROOT / "models" / "metrics.json"
MAX_LENGTH = 128  # matches the fine-tuning notebook

_model = None
_tokenizer = None
_id_to_label: Dict[int, str] = {}
load_error: Optional[str] = None


def available() -> bool:
    return _ensure_loaded()


def _ensure_loaded() -> bool:
    global _model, _tokenizer, _id_to_label, load_error
    if _model is not None:
        return True
    if load_error is not None:
        return False
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE).eval()
        _id_to_label = {int(i): name for i, name in _model.config.id2label.items()}
        if all(name.startswith("LABEL_") for name in _id_to_label.values()):
            # the original local export only has generic LABEL_0..4 names; the real mapping
            # was saved with the metrics (the Hub copy has it in config.json)
            label_to_id = json.loads(METRICS_PATH.read_text())["label_to_id"]
            _id_to_label = {v: k for k, v in label_to_id.items()}
        return True
    except Exception as e:  # missing deps or model files - report, don't crash the API
        load_error = f"{type(e).__name__}: {e}"
        return False


def normalize(text: str) -> str:
    # the CFPB training narratives were already lowercased with punctuation stripped;
    # bring raw incoming text closer to that distribution
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def classify(text: str) -> Dict:
    if not _ensure_loaded():
        raise RuntimeError(f"complaints classifier unavailable ({load_error})")
    import torch

    inputs = _tokenizer(normalize(text), truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # DistilBERT takes none; some tokenizer versions emit them anyway
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=-1)[0].tolist()
    scores = {_id_to_label[i]: round(p, 4) for i, p in enumerate(probs)}
    category = max(scores, key=scores.get)
    return {"category": category, "confidence": scores[category], "scores": scores}
