"""Agent + tools tests: the tools run against the real API (through the TestClient), Claude is a
scripted fake - no API key, no cost. The MCP test is skipped when `mcp` isn't installed."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from service import agent, agent_tools, guardrails
from service import app as app_module


@pytest.fixture()
def tools_via_testclient(client, monkeypatch):
    """Route the tools' HTTP calls into the in-process app instead of a network socket."""
    def call(method, path, params=None, body=None):
        r = client.request(method, path, params=params, json=body)
        if r.status_code >= 400:
            raise agent_tools.ToolError(f"API returned {r.status_code} for {path}: {r.json().get('detail')}")
        return r.json()
    monkeypatch.setattr(agent_tools, "_call", call)


# ---------------------------------------------------------------- tools
def test_find_at_risk_customers(tools_via_testclient):
    r = agent_tools.find_at_risk_customers(limit=5, min_probability=0.5, contract="Month-to-month")
    assert r["customers_scored"] > 0 and len(r["customers"]) <= 5
    probs = [c["churn_probability"] for c in r["customers"]]
    assert probs == sorted(probs, reverse=True) and all(p >= 0.5 for p in probs)
    assert all(c["contract"] == "Month-to-month" and c["top_risk_drivers"] for c in r["customers"])


def test_get_customer_and_unknown_id(tools_via_testclient):
    any_id = agent_tools.find_at_risk_customers(limit=1, min_probability=0)["customers"][0]["customer_id"]
    c = agent_tools.get_customer(any_id)
    assert c["customer_id"] == any_id and 0 <= c["churn_probability"] <= 1 and "Contract" in c["crm_fields"]
    with pytest.raises(agent_tools.ToolError, match="404"):
        agent_tools.get_customer("no-such-customer")


def test_check_drift_flags_contract_shift(tools_via_testclient):
    assert agent_tools.check_drift("contract_shift")["retrain_recommended"] is True
    assert agent_tools.check_drift("none")["retrain_recommended"] is False


def test_list_workflow_actions(tools_via_testclient, client):
    client.post("/actions", json={"customer_id": "c9", "action": "retention_email", "details": {}})
    r = agent_tools.list_workflow_actions(limit=5, action="retention_email")
    assert r["actions"] and all(a["action"] == "retention_email" for a in r["actions"])


def test_tool_definitions_come_from_signatures():
    defs = {d["name"]: d for d in agent.tool_definitions()}
    assert set(defs) == {fn.__name__ for fn in agent_tools.TOOLS}
    schema = defs["find_at_risk_customers"]["input_schema"]
    assert set(schema["properties"]) == {"limit", "min_probability", "contract"}
    assert schema["properties"]["limit"]["description"]  # from the docstring's Args section
    assert defs["get_customer"]["input_schema"]["required"] == ["customer_id"]


# ---------------------------------------------------------------- the agent loop
def _response(*blocks, stop="end_turn"):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                              cache_creation_input_tokens=0))


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool_use(name, args, id_="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


class FakeClaude:
    """Returns scripted responses in order and records every request."""
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        return self.responses.pop(0)


def test_agent_calls_tools_then_answers(tools_via_testclient):
    fake = FakeClaude(
        _response(_tool_use("get_platform_status", {}), stop="tool_use"),
        _response(_text("The model is healthy: CV ROC-AUC 0.84.")),
    )
    r = agent.ask("Is the model healthy?", client=fake)
    assert r["answer"].startswith("The model is healthy")
    assert [s["tool"] for s in r["steps"]] == ["get_platform_status"] and not r["steps"][0]["is_error"]
    assert {k: r["usage"][k] for k in ("input_tokens", "output_tokens", "model_calls")} == \
        {"input_tokens": 200, "output_tokens": 40, "model_calls": 2}
    assert r["usage"]["usd"] > 0
    # the tool result went back in the next request, matched to the tool_use id
    result_msg = fake.requests[1]["messages"][-1]
    assert result_msg["role"] == "user" and result_msg["content"][0]["tool_use_id"] == "toolu_1"
    assert "cv_roc_auc_mean" in result_msg["content"][0]["content"]
    assert {t["name"] for t in fake.requests[0]["tools"]} == {fn.__name__ for fn in agent_tools.TOOLS}


def test_agent_reports_tool_errors_to_claude(tools_via_testclient):
    fake = FakeClaude(
        _response(_tool_use("get_customer", {"customer_id": "nope"}), stop="tool_use"),
        _response(_text("I couldn't find that customer.")),
    )
    r = agent.ask("Tell me about customer nope", client=fake)
    assert r["steps"][0]["is_error"]
    assert fake.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_agent_stops_at_the_step_limit(tools_via_testclient, monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    fake = FakeClaude(*[_response(_tool_use("get_platform_status", {}), stop="tool_use") for _ in range(2)])
    r = agent.ask("loop forever", client=fake)
    assert r["stop_reason"] == "step_limit" and len(r["steps"]) == 2 and "ran out of steps" in r["answer"]


def test_agent_stops_at_the_cost_ceiling(tools_via_testclient):
    expensive = SimpleNamespace(input_tokens=20_000, output_tokens=3_000, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)  # $0.07 on Sonnet 5.5
    first = _response(_tool_use("get_platform_status", {}), stop="tool_use")
    first.usage, first.model = expensive, "claude-sonnet-5-5"
    fake = FakeClaude(first, _response(_text("never reached")))
    r = agent.ask("anything", client=fake, max_usd=0.05)
    assert r["stop_reason"] == "cost_limit" and len(fake.requests) == 1
    assert r["usage"]["usd"] == pytest.approx(0.07)


# ---------------------------------------------------------------- guardrails
def test_cost_is_priced_per_model_and_unknown_models_are_priced_high():
    u = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=0,
                        cache_creation_input_tokens=0)
    assert guardrails.cost_usd("claude-sonnet-5-5", u) == pytest.approx(2.0 + 1.0)
    assert guardrails.cost_usd("some-new-model", u) >= guardrails.cost_usd("claude-opus-5-5", u)


def _scope_client(text=None, stop="end_turn"):
    resp = SimpleNamespace(content=[_text(text)] if text else [], stop_reason=stop, model="claude-sonnet-5-5",
                           usage=SimpleNamespace(input_tokens=300, output_tokens=40, cache_read_input_tokens=0,
                                                 cache_creation_input_tokens=0))
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))


def test_scope_check_reads_the_verdict_and_fails_closed():
    ok = guardrails.check_scope("q", _scope_client('{"verdict": "on_topic", "reason": "churn"}'), "claude-sonnet-5-5")
    assert ok["verdict"] == "on_topic" and ok["usd"] > 0
    assert guardrails.check_scope("q", _scope_client(stop="refusal"), "m")["verdict"] == "harmful"
    assert guardrails.check_scope("q", _scope_client("not json"), "m")["verdict"] == "off_topic"


@pytest.fixture()
def endpoint(client, monkeypatch):
    """The /agent/ask endpoint with an in-memory event store, no scope check and a canned agent."""
    monkeypatch.setattr(app_module.rag, "drafting_configured", lambda: True)
    monkeypatch.setattr(guardrails, "_db", lambda: None)
    monkeypatch.setattr(guardrails, "_memory", [])
    monkeypatch.setattr(guardrails, "SCOPE_CHECK", False)
    monkeypatch.setattr(agent, "ask", lambda q, client=None, max_usd=0: {
        "question": q, "answer": "ok", "steps": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "model_calls": 1, "usd": 0.02}})
    return client


def limits(monkeypatch, **kw):
    lim = guardrails.Limits(**{f: 0 for f in guardrails.Limits.__dataclass_fields__} | kw)
    monkeypatch.setattr(guardrails, "LIMITS", lim)
    monkeypatch.setattr(guardrails.reserve, "__defaults__", (False, lim))
    monkeypatch.setattr(guardrails.status, "__defaults__", (lim,))


def ask(c, visitor="a"):
    return c.post("/agent/ask", json={"question": "which customers are at risk?"}, headers={"X-Visitor-Id": visitor})


def test_daily_question_limit(endpoint, monkeypatch):
    limits(monkeypatch, questions_per_day=2)
    assert ask(endpoint).status_code == ask(endpoint).status_code == 200
    r = ask(endpoint)
    assert r.status_code == 429 and "question budget" in r.json()["detail"]
    assert endpoint.get("/agent/status").json()["remaining_today"] == 0


def test_daily_dollar_budget_reserves_the_worst_case(endpoint, monkeypatch):
    # each question reserves $0.05 and really costs $0.02: after 3 ($0.06 spent) another
    # worst-case $0.05 would exceed $0.10, so the 4th is refused - the budget is never overshot
    limits(monkeypatch, usd_per_day=0.10, usd_per_question=0.05)
    assert [ask(endpoint).status_code for _ in range(4)] == [200, 200, 200, 429]
    assert endpoint.get("/agent/status").json()["spent_today_usd"] == pytest.approx(0.06)


def test_remaining_count_survives_float_division(endpoint, monkeypatch):
    limits(monkeypatch, usd_per_day=0.50, usd_per_question=0.05)  # 0.50 / 0.05 = 9.999... in floats
    assert endpoint.get("/agent/status").json()["remaining_today"] == 10


def test_per_visitor_hourly_limit(endpoint, monkeypatch):
    limits(monkeypatch, visitor_per_hour=1)
    assert ask(endpoint, "alice").status_code == 200
    assert ask(endpoint, "alice").status_code == 429
    assert ask(endpoint, "bob").status_code == 200  # other visitors aren't affected
    status = endpoint.get("/agent/status", headers={"X-Visitor-Id": "alice"}).json()
    assert status["remaining_for_you"] == 0


def test_failed_questions_dont_count(endpoint, monkeypatch):
    limits(monkeypatch, questions_per_day=2)

    def unavailable(q, client=None, max_usd=0):
        raise agent.AgentUnavailable("invalid Anthropic API key")
    monkeypatch.setattr(agent, "ask", unavailable)
    assert ask(endpoint).status_code == 503
    assert endpoint.get("/agent/status").json()["remaining_today"] == 2


def test_off_topic_questions_are_refused_before_the_agent_runs(endpoint, monkeypatch):
    limits(monkeypatch, visitor_per_day=5)
    monkeypatch.setattr(guardrails, "SCOPE_CHECK", True)
    monkeypatch.setattr(guardrails, "check_scope",
                        lambda q, client, model: {"verdict": "prompt_injection", "reason": "x", "usd": 0.001})
    monkeypatch.setattr(agent, "ask", lambda *a, **k: pytest.fail("the agent must not run"))
    body = ask(endpoint).json()
    assert body["blocked"] is True and "can't change how I work" in body["answer"]
    assert endpoint.get("/agent/status", headers={"X-Visitor-Id": "a"}).json()["remaining_for_you"] == 4
    assert guardrails._memory[-1]["outcome"] == "blocked_prompt_injection"
    assert guardrails._memory[-1]["visitor"] != "a"  # stored hashed


def test_public_demo_needs_limits_and_a_database(endpoint, monkeypatch):
    monkeypatch.setattr(app_module, "PUBLIC_DEMO", True)
    limits(monkeypatch)
    assert ask(endpoint).status_code == 403
    limits(monkeypatch, questions_per_day=5)
    r = ask(endpoint)  # limits can't be checked without the database: refuse
    assert r.status_code == 429 and "can't be checked" in r.json()["detail"]


def test_question_length_is_limited(endpoint):
    assert endpoint.post("/agent/ask", json={"question": "x" * 501}).status_code == 422


# ---------------------------------------------------------------- MCP
def test_mcp_server_exposes_the_same_read_only_tools():
    pytest.importorskip("mcp")
    from churn_mcp.server import server

    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {fn.__name__ for fn in agent_tools.TOOLS}
    assert all(t.annotations.read_only_hint for t in tools)
    assert set(next(t for t in tools if t.name == "check_drift").input_schema["properties"]) == {"scenario", "batch_size"}
