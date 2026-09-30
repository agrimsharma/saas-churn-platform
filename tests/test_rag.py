"""RAG tests against a real pgvector database (a separate churn_test DB, never the real index),
with a deterministic fake embedder and a fake Claude client - no torch, no API key, no cost.

Needs Postgres+pgvector at TEST_DATABASE_URL (default: the compose db on localhost:5433);
skipped when it isn't reachable. CI runs it as a service container."""
import hashlib
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from service import app as app_module
from service import rag

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://churn:churn@localhost:5433/churn_test")


def fake_embed(texts, batch_size=64):
    """Bag-of-words hashed into 384 dims: shared words -> similar vectors, deterministic."""
    out = np.zeros((len(texts), rag.EMBED_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % rag.EMBED_DIM] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


ROWS = [
    (1, "Debt collection", "Attempts to collect debt not owed", "Closed with explanation",
     "A debt collector keeps calling me about a medical debt that I already paid in full last year."),
    (2, "Debt collection", "Communication tactics", "Closed with non-monetary relief",
     "The collection agency calls my workplace every day about this debt and threatens me."),
    (3, "Mortgage", "Trouble during payment process", "Closed with explanation",
     "My mortgage servicer lost my loan modification paperwork and added late fees to my escrow."),
    (4, "Credit card", "Problem with a purchase shown on your statement", "Closed with monetary relief",
     "I was charged twice on my credit card statement for one purchase and the bank refuses a refund."),
]


@pytest.fixture(scope="module")
def db():
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    admin_url = TEST_DB.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = 'churn_test'").fetchone():
                admin.execute("CREATE DATABASE churn_test")
    except psycopg.OperationalError:
        pytest.skip("no Postgres/pgvector reachable at TEST_DATABASE_URL")
    mp = pytest.MonkeyPatch()
    mp.setattr(rag, "DATABASE_URL", TEST_DB)
    mp.setattr(rag, "embed", fake_embed)
    conn = rag.connect()
    rag.init_schema(conn)
    conn.execute("TRUNCATE complaints")
    emb = fake_embed([r[4] for r in ROWS])
    for (cid, product, issue, resp, text), e in zip(ROWS, emb):
        conn.execute(
            "INSERT INTO complaints (complaint_id, date_received, product, issue, company_response, "
            "narrative, embedding) VALUES (%s, '2023-05-01', %s, %s, %s, %s, %s)",
            (cid, product, issue, resp, text, e))
    yield conn
    conn.close()
    mp.undo()


def test_search_ranks_the_matching_complaint_first(db):
    hits = rag.search(db, fake_embed(["collector calling about a medical debt I already paid"])[0], k=3)
    assert hits[0]["complaint_id"] == 1
    assert [h["similarity"] for h in hits] == sorted((h["similarity"] for h in hits), reverse=True)


def test_search_product_filter_and_exclude(db):
    q = fake_embed(["collector calling about debt"])[0]
    assert {h["product"] for h in rag.search(db, q, k=4, product="Mortgage")} == {"Mortgage"}
    assert 1 not in [h["complaint_id"] for h in rag.search(db, q, k=4, exclude_id=1)]


def test_similar_endpoint(client, db):
    r = client.post("/complaints/similar", json={"text": "My mortgage servicer lost my loan modification papers", "k": 2})
    assert r.status_code == 200
    assert r.json()["similar"][0]["complaint_id"] == 3


def test_similar_endpoint_validates_input(client, db):
    assert client.post("/complaints/similar", json={"text": "too short"}).status_code == 422


def test_draft_without_api_key_is_503_not_500(client, db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = client.post("/complaints/draft", json={"text": "A debt collector keeps calling me about a paid debt."})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


class FakeClaude:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason, model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=900, output_tokens=150))


def test_draft_keeps_only_citations_to_supplied_cases(db):
    similar = rag.search(db, fake_embed(["debt collector calling"])[0], k=2)
    fake = FakeClaude({"reply": "We are reviewing...", "cited_complaint_ids": [1, 999],
                       "escalate": False, "escalation_reason": ""})
    draft = rag.draft_reply("A collector keeps calling about a debt I paid.", similar, client=fake)
    assert draft["cited_complaint_ids"] == [1]  # 999 was never retrieved -> dropped
    call = fake.calls[0]
    assert call["model"] == rag.CLAUDE_MODEL and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["schema"] == rag.DRAFT_SCHEMA
    assert "<new_complaint>" in call["messages"][0]["content"]


def test_draft_refusal_raises_unavailable(db):
    fake = FakeClaude({}, stop_reason="refusal")
    with pytest.raises(rag.DraftingUnavailable):
        rag.draft_reply("some complaint text here", [], client=fake)


def test_draft_endpoint_with_fake_claude(client, db, monkeypatch):
    fake = FakeClaude({"reply": "Thanks for reaching out...", "cited_complaint_ids": [4],
                       "escalate": True, "escalation_reason": "disputed charge"})
    real = rag.draft_reply
    monkeypatch.setattr(rag, "draft_reply", lambda text, similar: real(text, similar, client=fake))
    r = client.post("/complaints/draft", json={"text": "I was charged twice on my credit card for one purchase."})
    body = r.json()
    assert r.status_code == 200 and body["escalate"] is True and body["cited_complaint_ids"] == [4]
    assert body["similar"][0]["complaint_id"] == 4
