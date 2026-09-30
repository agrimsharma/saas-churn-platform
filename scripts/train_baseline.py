import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn_with_all_feedback.csv")
MLFLOW_DB = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}")
mlflow.set_experiment("churn-baseline-tabular")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# known Telco quirk: TotalCharges is blank string for tenure=0 (brand-new customers, no bill yet)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"Rows with unparseable TotalCharges (tenure=0, no bill yet): {df['TotalCharges'].isna().sum()}")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

y = (df["Churn"] == "Yes").astype(int)
print(f"Churn rate: {y.mean():.3f}")

# tabular-only baseline: drop identifiers and the text columns entirely (that's the point -
# this establishes what accuracy looks like BEFORE any text/NLP signal is added)
drop_cols = ["customerID", "Churn", "PromptInput", "CustomerFeedback"]
X = df.drop(columns=drop_cols)

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"Categorical: {categorical_cols}")
print(f"Numeric: {numeric_cols}")

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_and_log(name, model, X_tr, X_te, params, log_fn):
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        log_fn(model, name="model")

        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return metrics


logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
evaluate_and_log("logistic_regression", logreg, X_train_scaled, X_test_scaled,
                  {"model": "LogisticRegression", "class_weight": "balanced"},
                  mlflow.sklearn.log_model)

xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss", random_state=42,
)
evaluate_and_log("xgboost", xgb, X_train, X_test,
                  {"model": "XGBClassifier", "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
                  mlflow.xgboost.log_model)

print(f"\nMLflow runs logged to {MLFLOW_DB}")
print("View with: mlflow ui --backend-store-uri", f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}")
