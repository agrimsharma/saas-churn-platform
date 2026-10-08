"""The platform's tools, written once and served two ways:

  * to Claude through tool calling - service/agent.py (POST /agent/ask, the dashboard's Ask tab)
  * to any MCP client (Claude Desktop, Claude Code, IDEs) - churn_mcp/server.py

Each tool is a plain typed function: the signature and docstring become the tool's JSON schema
for both. Tools are thin clients of the HTTP API, so auth, validation and metrics stay in one
place, and the MCP server needs no ML dependencies. All tools are read-only: nothing here can
log actions, retrain the model or call Claude (which is billed).

Config: CHURN_API_URL (default http://127.0.0.1:8000) and CHURN_API_KEY (falls back to API_KEY).
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Literal, Optional

API_URL = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("CHURN_API_KEY") or os.environ.get("API_KEY")
# the "current customer book" the agent works on: a fixed sample of the CRM stand-in, so the
# same question gets the same customers
BOOK_SIZE = 500
BOOK_SEED = 7


class ToolError(Exception):
    """A tool failed in a way the model should see (bad input, API error) - not a crash."""


def _call(method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Dict:
    url = f"{API_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", e.reason)
        except ValueError:
            detail = e.reason
        raise ToolError(f"API returned {e.code} for {path}: {detail}") from e
    except urllib.error.URLError as e:
        raise ToolError(f"can't reach the churn API at {API_URL}: {e.reason}") from e


def _risk_tier(p: float) -> str:
    # the same thresholds as the n8n retention playbook
    return "high" if p >= 0.6 else "medium" if p >= 0.35 else "low"


def _scored(customers: List[Dict]) -> List[Dict]:
    preds = _call("POST", "/score", body={"customers": customers})["predictions"]
    by_id = {c["customerID"]: c for c in customers}
    out = []
    for p in preds:
        c = by_id[p["customer_id"]]
        out.append({
            "customer_id": p["customer_id"],
            "churn_probability": round(p["churn_probability"], 3),
            "risk_tier": _risk_tier(p["churn_probability"]),
            "contract": c["Contract"],
            "tenure_months": c["tenure"],
            "monthly_charges": c["MonthlyCharges"],
            "payment_method": c["PaymentMethod"],
            "internet_service": c["InternetService"],
            "top_risk_drivers": [f"{f['feature']} = {f['value']}" for f in p["top_risk_factors"]],
        })
    return out


def get_platform_status() -> Dict[str, Any]:
    """Health of the churn platform: the serving model's cross-validated ROC-AUC, when it was
    trained, and which optional features (complaint classifier, reply drafting) are available.
    Use this to answer questions about the model itself."""
    return _call("GET", "/health")


def find_at_risk_customers(limit: int = 10, min_probability: float = 0.5,
                           contract: Optional[Literal["Month-to-month", "One year", "Two year"]] = None
                           ) -> Dict[str, Any]:
    """Score the current customer book (500 customers) and return the ones most likely to churn,
    highest probability first, each with its risk tier and the top factors raising its risk
    (exact SHAP contributions). Probabilities are calibrated: 0.6 means about a 60% chance.

    Args:
        limit: how many customers to return (1-25).
        min_probability: only return customers at or above this churn probability (0-1).
        contract: optionally restrict to one contract type.
    """
    limit = max(1, min(int(limit), 25))
    book = _call("GET", "/demo/customers", params={"n": BOOK_SIZE, "seed": BOOK_SEED})["customers"]
    if contract:
        book = [c for c in book if c["Contract"] == contract]
    if not book:
        return {"customers_scored": 0, "matching": 0, "customers": []}
    scored = sorted(_scored(book), key=lambda r: -r["churn_probability"])
    matching = [r for r in scored if r["churn_probability"] >= min_probability]
    tiers = {t: sum(r["risk_tier"] == t for r in scored) for t in ("high", "medium", "low")}
    return {"customers_scored": len(scored), "risk_tier_counts": tiers,
            "matching": len(matching), "customers": matching[:limit]}


def get_customer(customer_id: str) -> Dict[str, Any]:
    """Look up one customer by ID (e.g. "7590-VHVEG"): their CRM fields, calibrated churn
    probability, risk tier and the top factors raising their risk.

    Args:
        customer_id: the customer's ID.
    """
    record = _call("GET", f"/demo/customers/{urllib.parse.quote(customer_id.strip())}")
    scored = _scored([record])[0]
    return {**scored, "crm_fields": {k: v for k, v in record.items() if k != "customerID"}}


def check_drift(scenario: Literal["none", "price_hike", "contract_shift", "both"] = "none",
                batch_size: int = 1000) -> Dict[str, Any]:
    """Run the drift check on a batch of current customers against the model's training data:
    PSI per feature, the share of model importance on drifted features, and whether retraining
    is recommended. The scenario injects the simulated drift the weekly workflow reacts to:
    "price_hike" (+25% monthly charges), "contract_shift" (70% of 1/2-year contracts move to
    month-to-month), or "both". Use "none" for the current, un-shifted customers.

    Args:
        scenario: which simulated shift to apply.
        batch_size: customers in the batch (100-3000).
    """
    n = max(100, min(int(batch_size), 3000))
    batch = _call("GET", "/demo/customers",
                  params={"n": n, "scenario": scenario, "include_labels": "true", "seed": BOOK_SEED})
    r = _call("POST", "/drift/check", body={"customers": batch["customers"], "churned": batch["churned"]})
    return {
        "scenario": scenario,
        "batch_size": n,
        "retrain_recommended": r["retrain_recommended"],
        "drifted_features": r["drifted_features"],
        "drifted_importance_mass": round(r["drifted_importance_mass"], 3),
        "importance_drift_threshold": r["importance_drift_threshold"],
        "psi_threshold": r["psi_threshold"],
        "top_feature_psi": dict(sorted(r["feature_psi"].items(), key=lambda kv: -kv[1])[:5]),
        "batch_roc_auc": r["current_roc_auc"],
        "roc_auc_drop": r.get("roc_auc_drop"),
    }


def classify_complaint(text: str) -> Dict[str, Any]:
    """Classify a consumer complaint into one of five product categories (credit card, credit
    reporting, debt collection, mortgages and loans, retail banking) with a confidence score,
    using the fine-tuned DistilBERT model.

    Args:
        text: the complaint text.
    """
    return _call("POST", "/classify", body={"text": text})


def find_similar_complaints(text: str, k: int = 5, product: Optional[str] = None) -> Dict[str, Any]:
    """Search 40,000 past CFPB consumer complaints for the ones most similar to this text
    (semantic search: bge-small embeddings in pgvector). Returns each case's product, issue,
    how the company responded, a similarity score, and the start of the narrative. Use it to
    see how similar problems were resolved.

    Args:
        text: the complaint or problem description (at least 20 characters).
        k: how many cases to return (1-10).
        product: optionally restrict to one CFPB product, e.g. "Debt collection".
    """
    k = max(1, min(int(k), 10))
    similar = _call("POST", "/complaints/similar", body={"text": text, "k": k, "product": product})["similar"]
    for s in similar:
        s["narrative"] = s["narrative"][:400] + ("…" if len(s["narrative"]) > 400 else "")
        s["similarity"] = round(s["similarity"], 3)
    return {"similar": similar}


def list_workflow_actions(limit: int = 20, action: Optional[str] = None) -> Dict[str, Any]:
    """The most recent actions the n8n workflows logged - retention calls and emails, routed
    complaints, model promotions or rejections, healthy drift checks - newest first.

    Args:
        limit: how many actions to return (1-100).
        action: optionally only one action type, e.g. "retention_call", "retention_email",
            "route_complaint", "model_promoted".
    """
    limit = max(1, min(int(limit), 100))
    actions = _call("GET", "/actions", params={"limit": 1000})["actions"]
    if action:
        actions = [a for a in actions if a["action"] == action]
    counts: Dict[str, int] = {}
    for a in actions:
        counts[a["action"]] = counts.get(a["action"], 0) + 1
    return {"counts_by_action": counts, "actions": actions[:limit]}


TOOLS = [get_platform_status, find_at_risk_customers, get_customer, check_drift,
         classify_complaint, find_similar_complaints, list_workflow_actions]
