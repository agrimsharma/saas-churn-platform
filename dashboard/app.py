"""Churn platform dashboard - the ops view locally, and the front page of the free public demo.

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
REPO = "https://github.com/agrimsharma/saas-churn-platform"

st.set_page_config(page_title="Churn Platform", layout="wide")


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", headers=HEADERS, timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def report(name):
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


try:
    health = api("GET", "/health")
except requests.RequestException as e:
    st.error(f"Can't reach the API at {API}: {e}")
    st.stop()
PUBLIC = health.get("public_demo", False)

st.title("Churn platform")
st.caption("Churn prediction with per-customer risk drivers, drift monitoring, and complaint triage with "
           f"retrieval-augmented drafting. [Source code]({REPO})")
if PUBLIC:
    st.info("**Free public demo.** Everything here runs live on free hosting, including the Claude agent in the "
            "**Ask the platform** tab (with a small daily question budget). Two parts run only in the full deployment "
            "(see the repo's screenshots): the **n8n workflows** that act on these predictions, and **live Claude "
            "reply drafting** - real example drafts are shown in the Complaint triage tab instead.")

m = health["churn_model"]
c1, c2, c3 = st.columns(3)
c1.metric("Churn model 5-fold CV ROC-AUC", f"{m['cv_roc_auc_mean']:.3f}" if m["cv_roc_auc_mean"] else "-")
c2.metric("Model trained", (m["trained_at"] or "-").replace("T", " ").rstrip("Z"))
c3.metric("Reply drafting", "live" if health.get("reply_drafting_configured") else "examples only")

tab_agent, tab_score, tab_complaints, tab_drift, tab_backtest, tab_actions = st.tabs(
    ["Ask the platform (agent)", "Churn scoring", "Complaint triage (RAG)", "Drift check", "Is retraining worth it?",
     "Workflow activity"])

# --- the Claude agent ----------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "Which 5 month-to-month customers are most likely to churn, and what offer would you make each?",
    "Is the model healthy? What happens if prices go up 25% and long contracts shift to month-to-month?",
    "A customer says a debt collector keeps calling about a medical bill insurance already paid. "
    "What category is that, and how were similar complaints resolved?",
    "Why is customer 7590-VHVEG at risk, and what has the platform already done about them?",
]


def visitor_headers():
    """Who's asking, for the API's per-visitor limits: the client address from the proxy header
    (or Streamlit's own view of it), else this browser session. The API only stores a hash."""
    ip = None
    try:
        ip = (st.context.headers.get("X-Forwarded-For") or "").split(",")[0].strip() or st.context.ip_address
    except Exception:
        pass
    if not ip:
        import uuid
        ip = st.session_state.setdefault("visitor_session", uuid.uuid4().hex)
    return {**HEADERS, "X-Visitor-Id": ip}

with tab_agent:
    st.write("Ask a question in plain English. Claude answers by **calling the platform's tools** - scoring "
             "customers, checking drift, searching past complaints, reading the workflow log - and shows each "
             "step. The same tools are available to Claude Desktop and Claude Code through the repo's **MCP server**.")
    try:
        status = requests.get(f"{API}/agent/status", headers=visitor_headers(), timeout=30).json()
    except (requests.RequestException, ValueError):
        status = {"configured": False}
    if not status.get("configured"):
        st.info("The agent isn't enabled here (no Anthropic API key). It runs locally and in the full deployment.")
    else:
        left_today, left_you = status.get("remaining_today"), status.get("remaining_for_you")
        budget = " · ".join(x for x in (
            f"{left_today} questions left today" if left_today is not None else "",
            f"{left_you} left for you" if left_you is not None else "") if x)
        st.caption((f"{budget} · " if budget else "") + f"Model: `{status['model']}` · tools: "
                   + ", ".join(f"`{t}`" for t in status["tools"]))
        if status.get("scope_check"):
            st.caption("Questions are screened first, so only questions about this platform are answered, and "
                       "are logged with a hashed visitor ID to prevent misuse.")
        choice = st.selectbox("Try an example, or write your own below", [""] + EXAMPLE_QUESTIONS)
        question = st.text_area("Question", value=choice, max_chars=500, height=90)
        out_of_budget = left_today == 0 or left_you == 0
        if st.button("Ask", type="primary", disabled=out_of_budget or len(question.strip()) < 3):
            with st.spinner("Claude is working through the tools..."):
                try:
                    r = requests.post(f"{API}/agent/ask", headers=visitor_headers(),
                                      json={"question": question.strip()}, timeout=300)
                    r.raise_for_status()
                    st.session_state["agent_result"] = r.json()
                    st.rerun()  # refresh the remaining-questions count
                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
                    st.session_state.pop("agent_result", None)
                    st.error(detail)
                except requests.RequestException as e:
                    st.error(f"The agent didn't answer: {e}")
        result = st.session_state.get("agent_result")
        if result and result.get("blocked"):
            st.warning(result["answer"])
        elif result:
            st.markdown(result["answer"])
            st.caption(f"{len(result['steps'])} tool call(s) · {result['usage']['model_calls']} model call(s) · "
                       f"{result['usage']['input_tokens']:,} in / {result['usage']['output_tokens']:,} out tokens · "
                       f"${result['usage'].get('usd', 0):.3f} · "
                       f"{result['seconds']} s · {result['model']}")
            for i, step in enumerate(result["steps"], 1):
                with st.expander(f"{'⚠️' if step['is_error'] else '🔧'} Step {i}: `{step['tool']}` "
                                 f"({step['seconds']} s)"):
                    st.markdown("**Claude called it with:**")
                    st.json(step["input"])
                    st.markdown("**The tool returned:**")
                    st.code(step["result_preview"], language="json")

# --- churn scoring -----------------------------------------------------------------------
with tab_score:
    st.write("Score a batch of customers (IBM Telco data, labels removed). Probabilities are calibrated; "
             "the drivers are exact XGBoost SHAP contributions mapped back to CRM fields.")
    n = st.slider("Customers", 5, 50, 15)
    if st.button("Score a random batch", type="primary"):
        batch = api("GET", "/demo/customers", params={"n": n})["customers"]
        preds = api("POST", "/score", json={"customers": batch})["predictions"]
        by_id = {c["customerID"]: c for c in batch}
        rows = [{
            "customer": p["customer_id"],
            "churn probability": p["churn_probability"],
            "risk": "high" if p["churn_probability"] >= 0.6 else "medium" if p["churn_probability"] >= 0.35 else "low",
            "contract": by_id[p["customer_id"]]["Contract"],
            "tenure (months)": by_id[p["customer_id"]]["tenure"],
            "monthly charges": by_id[p["customer_id"]]["MonthlyCharges"],
            "top risk drivers": ", ".join(f"{f['feature']} = {f['value']}" for f in p["top_risk_factors"]),
        } for p in preds]
        df = pd.DataFrame(rows).sort_values("churn probability", ascending=False)
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"churn probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.caption("In the full platform, n8n turns high risk into a call task and medium risk into a retention email, "
                   "with an offer picked from the customer's contract and payment method.")

# --- complaint triage ------------------------------------------------------------------
with tab_complaints:
    st.write("Classify a consumer complaint (DistilBERT fine-tuned on 162k CFPB complaints) and retrieve the most "
             "similar of 40,000 past complaints (bge-small embeddings in pgvector), with how each was resolved.")
    text = st.text_area("Complaint", height=110, value=(
        "A debt collector keeps calling me several times a day about a medical bill that my insurance already paid. "
        "They are now threatening to report it to the credit bureaus."))
    if st.button("Triage complaint", type="primary"):
        left, right = st.columns([1, 2])
        with left:
            try:
                c = api("POST", "/classify", json={"text": text})
                st.metric("Category", c["category"].replace("_", " "))
                st.caption(f"confidence {c['confidence']:.2f}")
                st.bar_chart(pd.Series(c["scores"]).sort_values())
            except requests.HTTPError as e:
                st.warning(f"Classifier unavailable: {e.response.json().get('detail', e)}")
        with right:
            try:
                sim = api("POST", "/complaints/similar", json={"text": text, "k": 5})["similar"]
                for s in sim:
                    with st.expander(f"{s['similarity']:.2f} · {s['product']} · {s['issue']} → {s['company_response']}"):
                        st.write(s["narrative"][:1200])
                        st.caption(f"CFPB complaint {s['complaint_id']} · {s['date_received']} · {s.get('company') or ''}")
            except requests.HTTPError as e:
                st.warning(f"Similar-complaint search unavailable: {e.response.json().get('detail', e)}")

    examples = report("example_drafts.json")
    if examples:
        st.subheader("Claude-drafted replies grounded in the retrieved cases")
        st.caption(f"Real outputs of POST /complaints/draft ({examples['model']}, structured output; citations are "
                   "restricted to retrieved cases). " + ("Shown as examples because live drafting is billed per call."
                                                         if PUBLIC else ""))
        for d in examples["drafts"]:
            with st.expander(("🚩 ESCALATE · " if d["escalate"] else "") + d["complaint"][:110] + "…"):
                st.markdown(f"**Complaint:** {d['complaint']}")
                st.markdown(f"**Draft reply:**\n\n{d['reply']}")
                st.caption(f"Escalate: {d['escalate']} - {d['escalation_reason']}  \n"
                           f"Cites past complaints {d['cited_complaint_ids']} · "
                           f"{d['input_tokens']} in / {d['output_tokens']} out tokens")

# --- drift check -----------------------------------------------------------------------
with tab_drift:
    st.write("Compare a batch of current customers against the model's training data: PSI per feature, weighted "
             "by how much the model relies on each feature. The scenarios inject the drift the n8n workflow reacts to.")
    col_a, col_b, col_c = st.columns(3)
    scenario = col_a.selectbox("Simulated scenario", ["none", "price_hike", "contract_shift", "both"], index=3)
    n = col_b.slider("Customers ", 100, 3000, 1000, step=100)
    with_labels = col_c.checkbox("Include outcomes (performance check)", value=True)
    if st.button("Run drift check", type="primary"):
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
            f"(threshold {r['importance_drift_threshold']:.0%})")
        if r["current_roc_auc"] is not None:
            st.caption(f"ROC-AUC on this batch: {r['current_roc_auc']:.3f} "
                       "(demo customers come from the training data, so this is in-sample)")
        st.bar_chart(pd.Series(r["feature_psi"], name="PSI"))
        st.caption(f"Population Stability Index per feature; > {r['psi_threshold']} counts as drifted.")

# --- retraining backtest on real time-stamped data ------------------------------------------
with tab_backtest:
    bt = report("retail_backtest.json")
    if not bt:
        st.info("Run `python scripts/retail_backtest.py` to generate the backtest.")
    else:
        st.write("Month-by-month replay of the monitoring + retraining loop on 2 years of real transactions "
                 "(UCI Online Retail II). Churn = no purchase in the next 90 days; models only train on snapshots "
                 "whose outcome was known at deployment time. True ROC-AUC is measured in hindsight.")
        monthly = pd.DataFrame(bt["monthly"])
        st.line_chart(monthly.pivot(index="month", columns="policy", values="true_roc_auc"))
        summary = pd.DataFrame(bt["summary"]).T[["mean_true_roc_auc", "min_true_roc_auc", "mean_top10_precision", "retrains"]]
        st.dataframe(summary, width="stretch")
        st.markdown("**Finding:** the model barely decays, so monthly retraining buys ~0.002 AUC. An earlier feature "
                    "set produced fake drift (lifetime totals grew just because the data starts in Dec 2009) that "
                    "would have caused 3 needless retrains.")

# --- what the workflows did ------------------------------------------------------------
with tab_actions:
    st.write("Actions the n8n workflows logged: retention calls and emails, routed complaints, model promotions."
             + (" This is a snapshot of a real local run - n8n isn't hosted in the free demo." if PUBLIC else ""))
    actions = pd.DataFrame(api("GET", "/actions", params={"limit": 1000})["actions"])
    if actions.empty:
        st.info("No actions logged yet - run a workflow in n8n (http://localhost:5678).")
    else:
        actions["logged_at"] = pd.to_datetime(actions["logged_at"], format="ISO8601")
        left, right = st.columns([1, 2])
        left.bar_chart(actions["action"].value_counts())
        details = pd.json_normalize(actions["details"]).add_prefix("details.")
        table = pd.concat([actions.drop(columns=["details"]), details], axis=1)
        right.dataframe(table.sort_values("logged_at", ascending=False), width="stretch", height=380)
