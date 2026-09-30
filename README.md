# SaaS Churn Platform

Churn prediction served behind a FastAPI service, with **n8n** workflows that turn predictions into actions:
- **Daily retention outreach:** score customers and create call or email tasks.
- **Complaint routing:** send each incoming complaint to the right team.
- **Drift-triggered retraining:** check for drift weekly and retrain, with a promotion gate.

A small Streamlit dashboard shows what the workflows did.

```
            ┌───────────────────── n8n (localhost:5678) ─────────────────────┐
 schedule → │ 01 retention   /demo/customers → /score → playbook → tasks + Slack │
 webhook  → │ 02 complaints  /classify → team routing → ticket                   │
 schedule → │ 03 drift       /drift/check → retrain? → /retrain (gated) → Slack  │
            └──────────────────────────────┬─────────────────────────────────┘
                                           │ HTTP + X-API-Key
 Streamlit dashboard (:8501) ──► FastAPI (:8000)
                                 XGBoost pipeline · isotonic calibration · SHAP drivers
                                 PSI drift vs training profile · DistilBERT complaints (opt.)
```

## Quick start

```bash
cp .env.example .env                 # set API_KEY; optionally SLACK_WEBHOOK_URL, WITH_NLP=true
docker compose up -d --build
docker compose exec n8n n8n import:workflow --separate --input=/workflows
```

| Service | URL |
|---|---|
| n8n | http://localhost:5678 (create the owner account, open a workflow, click **Execute workflow**) |
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

The complaint workflow must be **activated** before its webhook listens:

```bash
curl -X POST http://localhost:5678/webhook/complaint -H 'Content-Type: application/json' \
  -d '{"customer_id": "7590-VHVEG", "text": "A collector keeps calling about a debt I already paid."}'
```

### Without Docker

```bash
pip install -r requirements-dev.txt        # + requirements-nlp.txt for /classify
python scripts/train_scoring_model.py
uvicorn service.app:app --reload --port 8000
streamlit run dashboard/app.py
pytest tests
```

On macOS, xgboost needs OpenMP (`brew install libomp`).

**Data:** `data/raw/telco_churn_with_all_feedback.csv` (not committed). Any copy of the [IBM Telco churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) works via `CHURN_DATA_PATH=...`, and CI uses exactly that.

## Results

**Churn model** (XGBoost, 7,043 customers, 26.5% churn):

| Metric | Value |
|---|---|
| 5-fold CV ROC-AUC | **0.844 ± 0.014** |
| Held-out ROC-AUC | 0.850 |
| Recall / precision at the decision threshold | 0.80 / 0.52 |
| Brier score, raw → calibrated | 0.160 → **0.135** |
| Mean predicted churn, raw → calibrated (observed 0.265) | 0.403 → **0.269** |

How to read these:
- Accuracy is 0.75, against 0.735 for always predicting "stays". ROC-AUC and recall are the numbers that matter here.
- The class-weighted model ranks customers well, but its raw probabilities run high. An isotonic calibrator fitted on out-of-fold predictions fixes that, so the retention playbook's "≥ 0.6" means about 60%.
- Top model features: Contract (52%), InternetService (13%), OnlineSecurity (5%), TechSupport (5%).

**Drift detection** (`POST /drift/check`, 800-customer batches):

| Simulated shift | Drifted features (PSI > 0.2) | Model importance on them | Retrain? |
|---|---|---|---|
| none | – | 0% | no |
| +25% prices | MonthlyCharges | 1% | no |
| 70% of long contracts → month-to-month | Contract | 52% | **yes** |
| both | MonthlyCharges, Contract | 53% | **yes** |

A plain "share of columns drifted" metric treats the harmless price shift and the damaging contract shift alike. Weighting by model importance separates them.

**Complaints classifier** (DistilBERT fine-tuned on 162k real CFPB complaints, 5 products, Vast.ai RTX 3060 Ti):
- Accuracy **0.885**, macro-F1 **0.857**.
- A TF-IDF + logistic-regression baseline scores 0.846 / 0.818.

**Found and removed: label leakage.**
- The first text model predicted churn from customer feedback and scored **1.0 on every metric**.
- The feedback had been LLM-generated from a prompt containing `Churn: Yes/No`, so the text encoded the label.
- Masking the word "churn" (`scripts/prepare_text.py`) didn't help, because the sentiment itself carries the answer.
- The model was dropped. The scripts stay as a record (`prepare_text*.py`, `train_text_model.py`).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model version and CV ROC-AUC. No auth. |
| `POST /score` | `{"customers": [...]}` → calibrated churn probability + top 3 risk drivers per customer (exact XGBoost SHAP contributions, mapped back to CRM fields) |
| `POST /drift/check` | ≥ 100 customers (+ optional `churned` outcomes) → per-feature PSI, importance-weighted drift, retrain recommendation |
| `POST /retrain` | Retrains; **promotes only if 5-fold CV ROC-AUC ≥ current − 0.01**, else keeps the old model |
| `POST /classify` | `{"text": "..."}` → complaint category + confidence (needs `WITH_NLP=true`) |
| `GET /demo/customers` | CRM stand-in: random Telco rows; `scenario=price_hike\|contract_shift\|both` injects drift |
| `POST /actions`, `GET /actions` | Action log (`reports/actions.jsonl`), standing in for CRM writes |
| `GET /drift` | The original one-off Evidently report (`scripts/drift_monitoring.py`) |

## Workflows (`n8n/workflows/`, generated by `n8n/build_workflows.py`)

1. **Daily retention scoring** (weekdays, 08:00):
   - **High** risk (p ≥ 0.6, ~13% of customers): call task.
   - **Medium** risk (p ≥ 0.35, ~25%): retention email.
   - The offer comes from the customer's contract, support and payment method.
   - One Slack digest per run, including revenue at risk.
   - Thresholds and offers live in n8n, so business rules change without redeploying the model.
2. **Complaint routing:** webhook → DistilBERT category → owning team. Below 0.6 confidence goes to manual triage; legal or fraud keywords are flagged urgent.
3. **Weekly drift check & retrain:** pulls current customers, runs `/drift/check`, and if retraining is recommended calls `/retrain` and reports whether the candidate was promoted or rejected.

Slack steps are skipped when `SLACK_WEBHOOK_URL` is empty. To connect a real CRM, replace **Pull … customers** and the `/actions` nodes with HubSpot or Salesforce nodes.

## Limitations

- Demo customers come from the training data, so their scores and batch ROC-AUC are in-sample.
- `/retrain` retrains on the same static file. In production that file would be refreshed with newly labeled customers first.
- The drift scenarios are simulated; there is no real time-stamped production data.
- `mlflow.db` from the original experiments references Windows paths and a different project, so treat it as history. The serving model's metrics live in `models/churn_scoring/meta.json`.

## Layout

```
service/          FastAPI app, churn model + calibration, PSI drift, complaints classifier, Dockerfile
n8n/              workflow generator + importable workflows
dashboard/        Streamlit ops dashboard
tests/            API tests (pytest)
scripts/          training, original Evidently drift demo, text-leakage investigation, MLflow logging
notebooks/        GPU fine-tuning notebook (Vast.ai)
reports/          drift summary (simulated scenario)
```
