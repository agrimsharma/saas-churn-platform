"""Agent + tools tests: the tools run against the real API (through the TestClient), Claude is a
scripted fake - no API key, no cost. The MCP test is skipped when `mcp` isn't installed."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from service import agent, agent_tools
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
    assert r["usage"] == {"input_tokens": 200, "output_tokens": 40, "model_calls": 2}
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


# ---------------------------------------------------------------- the endpoint and the budget
@pytest.fixture()
def endpoint(client, monkeypatch):
    monkeypatch.setattr(app_module.rag, "drafting_configured", lambda: True)
    monkeypatch.setattr(agent, "_db", lambda: None)  # in-process budget counter
    monkeypatch.setattr(agent, "_memory_usage", {})
    monkeypatch.setattr(agent, "ask", lambda q: {"question": q, "answer": "ok", "steps": [], "usage":
                                                 {"input_tokens": 1, "output_tokens": 1, "model_calls": 1}})
    return client


def test_daily_budget_is_enforced(endpoint, monkeypatch):
    monkeypatch.setattr(agent, "DAILY_LIMIT", 2)
    assert endpoint.post("/agent/ask", json={"question": "hi there"}).json()["remaining_today"] == 1
    assert endpoint.post("/agent/ask", json={"question": "hi there"}).json()["remaining_today"] == 0
    r = endpoint.post("/agent/ask", json={"question": "hi there"})
    assert r.status_code == 429 and "budget" in r.json()["detail"]
    assert endpoint.get("/agent/status").json()["remaining_today"] == 0


def test_failed_questions_give_their_slot_back(endpoint, monkeypatch):
    monkeypatch.setattr(agent, "DAILY_LIMIT", 2)

    def unavailable(q):
        raise agent.AgentUnavailable("invalid Anthropic API key")
    monkeypatch.setattr(agent, "ask", unavailable)
    assert endpoint.post("/agent/ask", json={"question": "hi there"}).status_code == 503
    assert endpoint.get("/agent/status").json()["remaining_today"] == 2


def test_public_demo_needs_a_budget_and_a_database(endpoint, monkeypatch):
    monkeypatch.setattr(app_module, "PUBLIC_DEMO", True)
    monkeypatch.setattr(agent, "DAILY_LIMIT", 0)
    assert endpoint.post("/agent/ask", json={"question": "hi there"}).status_code == 403
    monkeypatch.setattr(agent, "DAILY_LIMIT", 5)
    assert endpoint.post("/agent/ask", json={"question": "hi there"}).status_code == 503  # no DB to count in


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
