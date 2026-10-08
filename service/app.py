"""Churn platform API - the HTTP surface n8n workflows call.

Run locally:  uvicorn service.app:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from service import agent, churn_model, complaints, drift, guardrails, rag

ROOT = Path(__file__).resolve().parents[1]
DRIFT_SUMMARY_PATH = ROOT / "reports" / "drift_summary.json"
ACTIONS_LOG_PATH = Path(os.environ.get("ACTIONS_LOG_PATH", ROOT / "reports" / "actions.jsonl"))
DEMO_ACTIONS_PATH = ROOT / "reports" / "demo_actions.jsonl"  # snapshot of a real n8n run
# free public deployment (Modal): nothing that changes shared state, and the only billed feature
# (the Claude agent) runs under a daily question budget
PUBLIC_DEMO = os.environ.get("PUBLIC_DEMO", "").lower() in ("1", "true", "yes")
API_KEY = os.environ.get("API_KEY")  # unset = no auth (local dev)
# a retrained model is only promoted if its CV ROC-AUC is at most this much worse
PROMOTION_TOLERANCE = float(os.environ.get("PROMOTION_TOLERANCE", "0.01"))

# Prometheus metrics (scraped at /metrics by the ServiceMonitor in the Helm chart)
REQUEST_SECONDS = Histogram("churn_api_request_seconds", "API latency by route", ["route", "method", "status"])
PREDICTIONS = Counter("churn_predictions_total", "Customers scored")
CHURN_PROB = Histogram("churn_probability", "Calibrated churn probability of scored customers",
                       buckets=[i / 10 for i in range(1, 11)])
DRIFT_MASS = Gauge("churn_drift_importance_mass", "Share of model importance on drifted features, last check")
RETRAIN_RECOMMENDED = Gauge("churn_retrain_recommended", "1 if the last drift check recommended a retrain")
DRAFTS = Counter("churn_reply_drafts_total", "Claude reply drafts", ["escalate"])
DRAFT_TOKENS = Counter("churn_reply_draft_tokens_total", "Claude tokens used for drafts", ["kind"])
# export every labelled series at 0 from startup: Prometheus' increase() can't see the first
# increment of a series that only appears once it's already at 1, so the first draft would read 0
for _label in ("true", "false"):
    DRAFTS.labels(_label)
for _label in ("input", "output"):
    DRAFT_TOKENS.labels(_label)
AGENT_QUESTIONS = Counter("churn_agent_questions_total", "Questions to the Claude agent", ["outcome"])
AGENT_TOOL_CALLS = Counter("churn_agent_tool_calls_total", "Tool calls made by the Claude agent", ["tool"])
AGENT_TOKENS = Counter("churn_agent_tokens_total", "Claude tokens used by the agent", ["kind"])
AGENT_SPEND = Counter("churn_agent_spend_usd_total", "Claude spend by the agent, incl. scope checks (USD)")
for _label in ("answered", "blocked", "limited", "unavailable", "error"):
    AGENT_QUESTIONS.labels(_label)
for _fn in agent.agent_tools.TOOLS:
    AGENT_TOOL_CALLS.labels(_fn.__name__)
for _label in ("input", "output"):
    AGENT_TOKENS.labels(_label)

_state: Dict[str, Any] = {"model": None}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = churn_model.load()
    if loaded is None:
        print("No compatible churn model found - training one (a few seconds)...")
        loaded = churn_model.train()
        churn_model.save(loaded)
    _state["model"] = loaded
    yield


app = FastAPI(title="SaaS Churn Platform API", version="0.2.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def time_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    if route is not None:  # skip /metrics and unknown paths (keeps label cardinality bounded)
        REQUEST_SECONDS.labels(route.path, request.method, str(response.status_code)).observe(time.perf_counter() - t0)
    return response


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def not_in_public_demo():
    if PUBLIC_DEMO:
        raise HTTPException(status_code=403, detail="disabled in the public demo (billed per call or "
                                                    "changes shared state) - see the full deployment")


class Customer(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None


class ScoreRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=5000)


class RetrainRequest(BaseModel):
    # newly labeled customers (e.g. the drifted batch once outcomes are known), added to the
    # training data; omit to retrain on the base data only
    customers: List[Customer] = Field(..., min_length=1, max_length=50000)
    churned: List[bool]


class DriftCheckRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=drift.MIN_BATCH, max_length=50000)
    # optional outcomes for these customers (e.g. from a later billing export) - enables the
    # performance check on top of the label-free feature drift check
    churned: Optional[List[bool]] = None


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class SimilarRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)
    k: int = Field(5, ge=1, le=20)
    product: Optional[str] = None  # restrict to one CFPB product, e.g. "Debt collection"


class AgentQuestion(BaseModel):
    question: str = Field(..., min_length=3, max_length=agent.MAX_QUESTION_CHARS)


class Action(BaseModel):
    customer_id: Optional[str] = None
    action: str
    details: Dict[str, Any] = {}
    source: str = "n8n"


@app.get("/health")
def health():
    meta = _state["model"]["meta"] if _state["model"] else {}
    return {
        "status": "ok",
        "churn_model": {
            "trained_at": meta.get("trained_at"),
            "cv_roc_auc_mean": meta.get("cv_roc_auc_mean"),
        },
        # don't force the (slow) DistilBERT load just for a health check
        "complaints_classifier_loaded": complaints._model is not None,
        "reply_drafting_configured": rag.drafting_configured() and not PUBLIC_DEMO,
        "public_demo": PUBLIC_DEMO,
    }


@app.post("/score", dependencies=[Depends(require_key)])
def score(req: ScoreRequest):
    records = [c.model_dump() for c in req.customers]
    with _lock:
        model = _state["model"]
    predictions = churn_model.score(model, records)
    PREDICTIONS.inc(len(predictions))
    for p in predictions:
        CHURN_PROB.observe(p["churn_probability"])
    return {"model_trained_at": model["meta"].get("trained_at"), "predictions": predictions}


@app.post("/classify", dependencies=[Depends(require_key)])
async def classify(req: ClassifyRequest):
    try:
        return await run_in_threadpool(complaints.classify, req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/drift/check", dependencies=[Depends(require_key)])
def drift_check(req: DriftCheckRequest):
    """Live drift check of a customer batch against the model's training data."""
    if req.churned is not None and len(req.churned) != len(req.customers):
        raise HTTPException(status_code=422, detail="churned must have one entry per customer")
    with _lock:
        model = _state["model"]
    X = churn_model.clean(pd.DataFrame([c.model_dump() for c in req.customers]))
    probs = model["pipeline"].predict_proba(X)[:, 1] if req.churned is not None else None
    result = drift.check(model["profile"], model["importances"], X, probs, req.churned,
                         model["meta"].get("cv_roc_auc_mean"))
    result["model_trained_at"] = model["meta"].get("trained_at")
    DRIFT_MASS.set(result["drifted_importance_mass"])
    RETRAIN_RECOMMENDED.set(int(result["retrain_recommended"]))
    return result


def _similar(req: SimilarRequest):
    try:
        conn = rag.connect()
    except Exception as e:  # db not running / not configured
        raise HTTPException(status_code=503, detail=f"complaint index unavailable: {type(e).__name__}")
    try:
        if rag.count(conn) == 0:
            raise HTTPException(status_code=503, detail="complaint index is empty - run scripts/build_complaint_index.py")
        return rag.search(conn, rag.embed([req.text])[0], k=req.k, product=req.product)
    finally:
        conn.close()


@app.post("/complaints/similar", dependencies=[Depends(require_key)])
async def similar_complaints(req: SimilarRequest):
    """Most similar past CFPB complaints, with how the company responded."""
    return {"similar": await run_in_threadpool(_similar, req)}


@app.post("/complaints/draft", dependencies=[Depends(require_key), Depends(not_in_public_demo)])
async def draft_complaint_reply(req: SimilarRequest):
    """Retrieve similar past complaints, then have Claude draft a reply grounded in them."""
    similar = await run_in_threadpool(_similar, req)
    try:
        draft = await run_in_threadpool(rag.draft_reply, req.text, similar)
    except rag.DraftingUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    DRAFTS.labels(str(draft["escalate"]).lower()).inc()
    DRAFT_TOKENS.labels("input").inc(draft["usage"]["input_tokens"])
    DRAFT_TOKENS.labels("output").inc(draft["usage"]["output_tokens"])
    return {**draft, "similar": [{k: c[k] for k in ("complaint_id", "product", "issue", "company_response", "similarity")}
                                 for c in similar]}


@app.get("/drift", dependencies=[Depends(require_key)])
def drift_report():
    """The original one-off Evidently report (simulated scenario, scripts/drift_monitoring.py)."""
    if not DRIFT_SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="no drift summary - run scripts/drift_monitoring.py")
    summary = json.loads(DRIFT_SUMMARY_PATH.read_text())
    summary["generated_at"] = datetime.fromtimestamp(
        DRIFT_SUMMARY_PATH.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    return summary


@app.post("/retrain", dependencies=[Depends(require_key), Depends(not_in_public_demo)])
async def retrain(req: Optional[RetrainRequest] = None):
    """Retrain on the base data plus any newly labeled customers; promote only if CV ROC-AUC holds up."""
    recent = None
    if req is not None:
        if len(req.churned) != len(req.customers):
            raise HTTPException(status_code=422, detail="churned must have one entry per customer")
        recent = (pd.DataFrame([c.model_dump() for c in req.customers]), req.churned)
    candidate = await run_in_threadpool(churn_model.train, recent)
    old_auc = _state["model"]["meta"].get("cv_roc_auc_mean", 0.0)
    new_auc = candidate["meta"]["cv_roc_auc_mean"]
    promoted = new_auc >= old_auc - PROMOTION_TOLERANCE
    if promoted:
        churn_model.save(candidate)
        with _lock:
            _state["model"] = candidate
    return {
        "promoted": promoted,
        "previous_cv_roc_auc": old_auc,
        "candidate_cv_roc_auc": new_auc,
        "new_labeled_rows": candidate["meta"]["n_recent_rows"],
        "candidate_holdout_metrics": candidate["meta"]["holdout_metrics"],
    }


@app.get("/demo/customers", dependencies=[Depends(require_key)])
def demo_customers(
    n: int = Query(25, ge=1, le=5000),
    seed: Optional[int] = None,
    scenario: Literal["none", "price_hike", "contract_shift", "both"] = "none",
    include_labels: bool = False,
):
    """Stand-in for a CRM "active customers" export: random Telco rows, labels removed.

    scenario applies the simulated drift from scripts/drift_monitoring.py (+25% prices,
    70% of 1/2-year contracts moved to month-to-month) so the drift workflow has
    something to detect. include_labels returns true outcomes as `churned`, for the
    drift check's performance test.
    """
    df = pd.read_csv(churn_model.DATA_PATH)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample["TotalCharges"] = pd.to_numeric(sample["TotalCharges"], errors="coerce").fillna(0)
    if scenario != "none":
        sample = drift.simulate(sample, scenario, seed=seed or 42)
    out = {"scenario": scenario,
           "customers": sample[["customerID"] + churn_model.FEATURES].to_dict(orient="records")}
    if include_labels:
        out["churned"] = (sample["Churn"] == "Yes").tolist()
    return out


@app.get("/demo/customers/{customer_id}", dependencies=[Depends(require_key)])
def demo_customer(customer_id: str):
    """One customer from the CRM stand-in, by ID (label removed)."""
    df = pd.read_csv(churn_model.DATA_PATH)
    row = df[df["customerID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"no customer with ID {customer_id}")
    row = row.iloc[[0]].copy()
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce").fillna(0)
    return row[["customerID"] + churn_model.FEATURES].to_dict(orient="records")[0]


@app.post("/actions", dependencies=[Depends(require_key), Depends(not_in_public_demo)])
def log_action(action: Action):
    """Local action log - where a real deployment would write to the CRM."""
    entry = {"logged_at": datetime.now(timezone.utc).isoformat(), **action.model_dump()}
    ACTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, ACTIONS_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


@app.get("/actions", dependencies=[Depends(require_key)])
def list_actions(limit: int = Query(50, ge=1, le=1000)):
    path = ACTIONS_LOG_PATH if ACTIONS_LOG_PATH.exists() else DEMO_ACTIONS_PATH
    if not path.exists():
        return {"actions": []}
    lines = path.read_text().splitlines()[-limit:]
    return {"actions": [json.loads(line) for line in reversed(lines)]}


def _visitor(request: Request, x_visitor_id: Optional[str]) -> str:
    # the dashboard forwards the visitor's address (the API itself is private on the free demo);
    # direct API callers are identified by their own address
    return guardrails.visitor_id(x_visitor_id or (request.client.host if request.client else None))


def _agent_enabled() -> bool:
    # the public demo only runs the agent with limits configured
    return rag.drafting_configured() and (not PUBLIC_DEMO or guardrails.LIMITS.any())


@app.get("/agent/status", dependencies=[Depends(require_key)])
def agent_status(request: Request, x_visitor_id: Optional[str] = Header(default=None)):
    return {
        "configured": _agent_enabled(),
        "model": agent.AGENT_MODEL,
        "tools": [fn.__name__ for fn in agent.agent_tools.TOOLS],
        "scope_check": guardrails.SCOPE_CHECK,
        **guardrails.status(_visitor(request, x_visitor_id)),
    }


@app.post("/agent/ask", dependencies=[Depends(require_key)])
def agent_ask(req: AgentQuestion, request: Request, x_visitor_id: Optional[str] = Header(default=None)):
    """Answer a question with the Claude agent, which calls the platform's read-only tools.
    Guardrails first (service/guardrails.py): usage and spending limits, then a scope check that
    refuses off-topic, prompt-injection and harmful questions before the agent runs."""
    import anthropic

    if PUBLIC_DEMO and not guardrails.LIMITS.any():
        raise HTTPException(status_code=403, detail="the agent is disabled in the public demo")
    if not rag.drafting_configured():
        AGENT_QUESTIONS.labels("unavailable").inc()
        raise HTTPException(status_code=503, detail="the agent is disabled: set ANTHROPIC_API_KEY")
    question = req.question.strip()
    try:
        event = guardrails.reserve(_visitor(request, x_visitor_id), question, require_db=PUBLIC_DEMO)
    except guardrails.LimitReached as e:
        AGENT_QUESTIONS.labels("limited").inc()
        raise HTTPException(status_code=429, detail=str(e))

    spent = 0.0
    try:
        client = anthropic.Anthropic()
        if guardrails.SCOPE_CHECK:
            scope = guardrails.check_scope(question, client, agent.AGENT_MODEL)
            spent += scope["usd"]
            if scope["verdict"] != "on_topic":
                guardrails.settle(event, f"blocked_{scope['verdict']}", spent)
                AGENT_QUESTIONS.labels("blocked").inc()
                AGENT_SPEND.inc(spent)
                return {"question": question, "answer": guardrails.REFUSALS[scope["verdict"]], "blocked": True,
                        "steps": [], "usage": {"input_tokens": 0, "output_tokens": 0, "model_calls": 0,
                                               "usd": round(spent, 5)},
                        "model": agent.AGENT_MODEL, "seconds": 0}
        result = agent.ask(question, client=client, max_usd=guardrails.LIMITS.usd_per_question)
    except (agent.AgentUnavailable, anthropic.AuthenticationError, anthropic.RateLimitError,
            anthropic.APIConnectionError) as e:
        guardrails.settle(event, "failed" if not spent else "error", spent)
        AGENT_QUESTIONS.labels("unavailable").inc()
        detail = str(e) if isinstance(e, agent.AgentUnavailable) else "the Claude API is unavailable - try again later"
        raise HTTPException(status_code=503, detail=detail)
    except Exception:
        guardrails.settle(event, "failed" if not spent else "error", spent)
        AGENT_QUESTIONS.labels("error").inc()
        raise
    spent += result["usage"]["usd"]
    result["usage"]["usd"] = round(spent, 5)
    guardrails.settle(event, "answered", spent)
    AGENT_QUESTIONS.labels("answered").inc()
    AGENT_SPEND.inc(spent)
    for step in result["steps"]:
        AGENT_TOOL_CALLS.labels(step["tool"]).inc()
    AGENT_TOKENS.labels("input").inc(result["usage"]["input_tokens"])
    AGENT_TOKENS.labels("output").inc(result["usage"]["output_tokens"])
    result["blocked"] = False
    return result
