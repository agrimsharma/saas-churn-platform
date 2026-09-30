"""Train and save the serving churn pipeline (models/churn_scoring/).

The API also trains one automatically on first start if none exists; run this
to (re)build it by hand and see the evaluation numbers.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from service import churn_model  # noqa: E402

result = churn_model.train()
churn_model.save(result)

meta = result["meta"]
print(f"Trained on {meta['n_rows']} rows (churn rate {meta['churn_rate']:.3f})")
print("Held-out split (20%):")
for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
    print(f"  {k}: {meta['holdout_metrics'][k]:.4f}")
m = meta["holdout_metrics"]
print(f"Calibration (held-out): Brier {m['brier_raw']:.4f} raw -> {m['brier_calibrated']:.4f} calibrated; "
      f"mean predicted {m['mean_predicted_raw']:.3f} raw / {m['mean_predicted_calibrated']:.3f} calibrated "
      f"vs observed {m['observed_churn_rate']:.3f}")
print(f"Top features: {meta['top_features']}")
print(f"5-fold CV ROC-AUC: {meta['cv_roc_auc_mean']:.4f} +/- {meta['cv_roc_auc_std']:.4f}")
print(f"Saved to {churn_model.PIPELINE_PATH}")
