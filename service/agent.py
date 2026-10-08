"""The "Ask the platform" agent: Claude that answers questions by calling the platform's tools.

The loop: send the question with the tool definitions → Claude replies with tool_use blocks →
run those tools against the API → send the results back → repeat until Claude answers in text
(or the step limit is reached). A hand-written loop rather than the SDK's beta tool runner, so
every step can be recorded for the UI, capped, and counted in metrics.

Each question is answered independently (no multi-turn memory): cheaper, simpler to cap, and
the conversation history stays append-only within one question.

The daily question budget lives in Postgres (agent_usage), so it holds across restarts and
across containers; without a database (local dev) it falls back to an in-process counter.
"""
import json
import os
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from service import agent_tools, rag

AGENT_MODEL = os.environ.get("CLAUDE_AGENT_MODEL", rag.CLAUDE_MODEL)
AGENT_EFFORT = os.environ.get("CLAUDE_AGENT_EFFORT", "medium")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "6"))          # model calls per question
DAILY_LIMIT = int(os.environ.get("AGENT_DAILY_LIMIT", "0"))       # questions per UTC day; 0 = unlimited
MAX_QUESTION_CHARS = 500
RESULT_PREVIEW_CHARS = 1500

SYSTEM_PROMPT = """You are the analyst assistant for a churn-prediction platform used by a \
subscription business's retention and support teams. You answer questions about customers, \
churn risk, model health, drift, complaints and what the automated workflows did.

Use the tools to get real data before answering; never invent customers, numbers or IDs. If \
the tools can't answer the question, say so plainly. The customer data is a demo dataset (IBM \
Telco), and the complaints are real public CFPB complaints.

When you recommend a retention action, base it on the customer's actual risk drivers (contract, \
tenure, payment method, services) and say which driver each suggestion addresses. Keep answers \
short and scannable: lead with the answer, then the supporting numbers. Use markdown bullets or \
a small table when listing customers."""


class AgentUnavailable(Exception):
    """No credentials, or the API rejected the request in a way retrying won't fix."""


class BudgetExhausted(Exception):
    """Today's question budget is used up."""


# ---------------------------------------------------------------- daily budget
_memory_usage: Dict[str, int] = {}
_memory_lock = threading.Lock()

USAGE_SCHEMA = """CREATE TABLE IF NOT EXISTS agent_usage (
    day        DATE PRIMARY KEY,
    questions  INTEGER NOT NULL
)"""


def _db():
    try:
        conn = rag.connect()
    except Exception:
        return None
    conn.execute(USAGE_SCHEMA)
    return conn


def used_today() -> int:
    conn = _db()
    if conn is None:
        return _memory_usage.get(str(date.today()), 0)
    with conn:
        row = conn.execute("SELECT questions FROM agent_usage WHERE day = CURRENT_DATE").fetchone()
    return row[0] if row else 0


def take_slot(require_db: bool = False) -> int:
    """Reserve one question in today's budget; returns how many remain. Atomic in Postgres:
    the conditional upsert only increments while below the limit, so concurrent requests
    can't overshoot it."""
    if DAILY_LIMIT <= 0:
        return -1
    conn = _db()
    if conn is None:
        if require_db:
            raise AgentUnavailable("the question budget needs the database, which isn't reachable")
        with _memory_lock:
            key = str(date.today())
            if _memory_usage.get(key, 0) >= DAILY_LIMIT:
                raise BudgetExhausted
            _memory_usage[key] = _memory_usage.get(key, 0) + 1
            return DAILY_LIMIT - _memory_usage[key]
    with conn:
        row = conn.execute(
            "INSERT INTO agent_usage (day, questions) VALUES (CURRENT_DATE, 1) "
            "ON CONFLICT (day) DO UPDATE SET questions = agent_usage.questions + 1 "
            "WHERE agent_usage.questions < %s RETURNING questions", (DAILY_LIMIT,)).fetchone()
    if row is None:
        raise BudgetExhausted
    return DAILY_LIMIT - row[0]


def release_slot() -> None:
    """Give a reserved question back - the call failed before Claude answered (bad key,
    rate limit), so the visitor shouldn't lose part of today's budget."""
    if DAILY_LIMIT <= 0:
        return
    conn = _db()
    if conn is None:
        with _memory_lock:
            key = str(date.today())
            _memory_usage[key] = max(_memory_usage.get(key, 0) - 1, 0)
        return
    with conn:
        conn.execute("UPDATE agent_usage SET questions = questions - 1 "
                     "WHERE day = CURRENT_DATE AND questions > 0")


# ---------------------------------------------------------------- the agent loop
def tool_definitions() -> List[Dict]:
    """Claude tool schemas, generated from the tool functions' signatures and docstrings."""
    from anthropic import beta_tool

    return [beta_tool(fn).to_dict() for fn in agent_tools.TOOLS]


_TOOLS_BY_NAME: Dict[str, Callable] = {fn.__name__: fn for fn in agent_tools.TOOLS}


def run_tool(name: str, args: Dict[str, Any]) -> tuple:
    """(result text, is_error). Errors go back to Claude as tool results so it can recover."""
    fn = _TOOLS_BY_NAME.get(name)
    if fn is None:
        return f"unknown tool: {name}", True
    try:
        return json.dumps(fn(**args), default=str), False
    except agent_tools.ToolError as e:
        return str(e), True
    except TypeError as e:  # wrong or missing arguments
        return f"invalid arguments for {name}: {e}", True


def ask(question: str, client=None, on_tool: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    import anthropic

    if client is None:
        if not rag.drafting_configured():
            raise AgentUnavailable("the agent is disabled: set ANTHROPIC_API_KEY")
        client = anthropic.Anthropic()

    tools = tool_definitions()
    messages: List[Dict] = [{"role": "user", "content": question}]
    steps: List[Dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
    answer, stop, model = None, None, AGENT_MODEL
    t0 = time.perf_counter()

    for _ in range(MAX_STEPS):
        try:
            response = client.beta.messages.create(
                model=AGENT_MODEL,
                max_tokens=8000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # re-run on Anthropic's recommended model if a classifier declines
                output_config={"effort": AGENT_EFFORT},
                cache_control={"type": "ephemeral"},  # tools + system are identical on every call
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except anthropic.AuthenticationError as e:
            raise AgentUnavailable("invalid Anthropic API key") from e
        except anthropic.RateLimitError as e:
            raise AgentUnavailable("rate limited by the Claude API - retry in a minute") from e
        except anthropic.APIConnectionError as e:
            raise AgentUnavailable("can't reach the Claude API") from e

        usage["model_calls"] += 1
        usage["input_tokens"] += response.usage.input_tokens + (response.usage.cache_read_input_tokens or 0) \
            + (response.usage.cache_creation_input_tokens or 0)
        usage["output_tokens"] += response.usage.output_tokens
        model, stop = response.model, response.stop_reason
        # append the whole content (thinking + text + tool_use), never just the text
        messages.append({"role": "assistant", "content": response.content})

        if stop == "refusal":
            answer = "I can't help with that request."
            break
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if stop != "tool_use" or not tool_uses:
            answer = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
            break

        results = []
        for block in tool_uses:  # Claude may ask for several tools at once
            if on_tool:
                on_tool(block.name)
            started = time.perf_counter()
            text, is_error = run_tool(block.name, dict(block.input))
            steps.append({
                "tool": block.name,
                "input": dict(block.input),
                "is_error": is_error,
                "seconds": round(time.perf_counter() - started, 2),
                "result_preview": text[:RESULT_PREVIEW_CHARS] + ("…" if len(text) > RESULT_PREVIEW_CHARS else ""),
            })
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text,
                            **({"is_error": True} if is_error else {})})
        # all results for one turn go back in a single user message
        messages.append({"role": "user", "content": results})
    else:
        stop = "step_limit"

    if not answer:
        answer = ("I ran out of steps before finishing - try a narrower question."
                  if stop == "step_limit" else "I couldn't produce an answer - try rephrasing.")
    return {
        "question": question,
        "answer": answer,
        "steps": steps,
        "stop_reason": stop,
        "model": model,
        "usage": usage,
        "seconds": round(time.perf_counter() - t0, 1),
        "answered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
