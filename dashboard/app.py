"""Ops dashboard: model health, what the n8n workflows did, and an on-demand drift check.

Run:  streamlit run dashboard/app.py   (CHURN_API_URL / CHURN_API_KEY env vars)
"""
import json
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("CHURN_API_URL", "http://localhost:8000")
REPORTS = os.environ.get("REPORTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports"))
HEADERS = {"X-API-Key": os.environ["CHURN_API_KEY"]} if os.environ.get("CHURN_API_KEY") else {}

st.set_page_config(page_title="Churn Platform", layout="wide")
st.title("Churn platform")


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", headers=HEADERS, timeout=60, **kw)
    r.raise_for_status()
    return r.json()


try:
    health = api("GET", "/health")
except requests.RequestException as e:
    st.error(f"Can't reach the API at {API}: {e}")
    st.stop()

m = health["churn_model"]
c1, c2, c3 = st.columns(3)
c1.metric("Model trained", (m["trained_at"] or "-").replace("T", " ").rstrip("Z"))
c2.metric("5-fold CV ROC-AUC", f"{m['cv_roc_auc_mean']:.3f}" if m["cv_roc_auc_mean"] else "-")
c3.metric("Complaints classifier", "loaded" if health["complaints_classifier_loaded"] else "not loaded")

# --- what the workflows did --------------------------------------------------------------
st.subheader("Workflow actions")
actions = pd.DataFrame(api("GET", "/actions", params={"limit": 1000})["actions"])
if actions.empty:
    st.info("No actions logged yet - run a workflow in n8n (http://localhost:5678).")
else:
    actions["logged_at"] = pd.to_datetime(actions["logged_at"])
    left, right = st.columns([1, 2])
    left.bar_chart(actions["action"].value_counts())
    details = pd.json_normalize(actions["details"]).add_prefix("details.")
    table = pd.concat([actions.drop(columns=["details"]), details], axis=1)
    right.dataframe(table.sort_values("logged_at", ascending=False), width="stretch", height=320)

# --- on-demand drift check ---------------------------------------------------------------
st.subheader("Drift check")
col_a, col_b, col_c = st.columns(3)
scenario = col_a.selectbox("Simulated scenario", ["none", "price_hike", "contract_shift", "both"], index=3)
n = col_b.slider("Customers", 100, 3000, 1000, step=100)
with_labels = col_c.checkbox("Include outcomes (performance check)", value=True)

if st.button("Run drift check"):
    batch = api("GET", "/demo/customers",
                params={"n": n, "scenario": scenario, "include_labels": str(with_labels).lower()})
    body = {"customers": batch["customers"]}
    if with_labels:
        body["churned"] = batch["churned"]
    r = api("POST", "/drift/check", json=body)

    verdict = "Retrain recommended" if r["retrain_recommended"] else "Model healthy"
    (st.error if r["retrain_recommended"] else st.success)(
        f"{verdict} - drifted: {', '.join(r['drifted_features']) or 'none'}; "
        f"{r['drifted_importance_mass']:.0%} of model importance on drifted features "
        f"(threshold {r['importance_drift_threshold']:.0%})"
    )
    if r["current_roc_auc"] is not None:
        st.caption(f"ROC-AUC on this batch: {r['current_roc_auc']:.3f} "
                   "(demo customers come from the training data, so this is in-sample)")
    psi = pd.Series(r["feature_psi"], name="PSI")
    st.bar_chart(psi)
    st.caption(f"Population Stability Index per feature; > {r['psi_threshold']} counts as drifted.")

# --- retraining backtest on real time-stamped data (scripts/retail_backtest.py) ------------
st.subheader("Is retraining worth it? Backtest on 2 years of real transactions")
path = os.path.join(REPORTS, "retail_backtest.json")
if not os.path.exists(path):
    st.info("Run `python scripts/retail_backtest.py` to generate the backtest.")
else:
    with open(path) as f:
        bt = json.load(f)
    monthly = pd.DataFrame(bt["monthly"])
    st.line_chart(monthly.pivot(index="month", columns="policy", values="true_roc_auc"))
    summary = pd.DataFrame(bt["summary"]).T[["mean_true_roc_auc", "min_true_roc_auc", "mean_top10_precision", "retrains"]]
    st.dataframe(summary, width="stretch")
    st.caption("UCI Online Retail II, monthly snapshots; churn = no purchase in the next 90 days. Models only "
               "train on snapshots whose outcome was known at deployment time. True ROC-AUC is measured in hindsight.")
