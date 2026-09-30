"""Similar-complaint retrieval (pgvector) + grounded reply drafting (Claude).

Retrieval works without any API key. Drafting needs ANTHROPIC_API_KEY (or an `ant auth login`
profile); without one, /complaints/draft returns 503 and everything else keeps working.
"""
import os
from typing import Dict, List, Optional

import numpy as np

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
MAX_SEQ_LENGTH = 256
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://churn:churn@localhost:5432/churn")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id      BIGINT PRIMARY KEY,
    date_received     DATE NOT NULL,
    product           TEXT NOT NULL,
    sub_product       TEXT,
    issue             TEXT,
    sub_issue         TEXT,
    company           TEXT,
    company_response  TEXT,
    timely_response   BOOLEAN,
    state             TEXT,
    narrative         TEXT NOT NULL,
    embedding         vector({EMBED_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS complaints_embedding_hnsw
    ON complaints USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS complaints_product ON complaints (product);
"""

_embedder = None


def embedder():
    """Lazy: sentence-transformers pulls in torch, only present in the NLP image."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        # first 256 tokens carry the gist of a complaint; ~2x faster than the 512 default on CPU.
        # Applies to indexing and queries alike, so the two stay consistent.
        _embedder.max_seq_length = MAX_SEQ_LENGTH
    return _embedder


def embed(texts: List[str], batch_size: int = 64) -> np.ndarray:
    # symmetric task (complaint vs complaint), so no bge query instruction prefix
    return embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn) -> None:
    conn.execute(SCHEMA)


def count(conn) -> int:
    return conn.execute("SELECT count(*) FROM complaints").fetchone()[0]


COLUMNS = ["complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
           "company", "company_response", "timely_response", "state", "narrative"]


def search(conn, query_embedding: np.ndarray, k: int = 5, product: Optional[str] = None,
           exclude_id: Optional[int] = None) -> List[Dict]:
    where, params = [], {"q": query_embedding, "k": k}
    if product:
        where.append("product = %(product)s")
        params["product"] = product
    if exclude_id is not None:
        where.append("complaint_id <> %(exclude)s")
        params["exclude"] = exclude_id
    sql = (f"SELECT {', '.join(COLUMNS)}, 1 - (embedding <=> %(q)s) AS similarity FROM complaints "
           f"{'WHERE ' + ' AND '.join(where) if where else ''} "
           "ORDER BY embedding <=> %(q)s LIMIT %(k)s")
    rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(zip(COLUMNS + ["similarity"], r))
        d["date_received"] = d["date_received"].isoformat()
        d["similarity"] = round(float(d["similarity"]), 4)
        out.append(d)
    return out


# --- drafting -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You draft first replies to consumer financial complaints for a support team. \
A human agent reviews every draft before it is sent.

You are given the new complaint and a few similar past complaints, each with how the company \
responded. Use the past cases to understand what usually resolves this kind of issue and what \
the consumer will need to provide - but write about the new complaint only. Never state facts \
about the consumer's account that the new complaint doesn't contain, and never promise refunds, \
credits or outcomes; say what the team will review or do next.

Write a reply of 80-160 words: acknowledge the specific problem, state the next step, and list \
any documents or details needed. Cite the past complaint IDs whose handling informed the reply. \
Set escalate to true when the complaint alleges fraud, identity theft, discrimination, legal \
action, or a regulator, or when the past cases show this issue is usually not resolved at first \
contact."""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "cited_complaint_ids": {"type": "array", "items": {"type": "integer"}},
        "escalate": {"type": "boolean"},
        "escalation_reason": {"type": "string"},
    },
    "required": ["reply", "cited_complaint_ids", "escalate", "escalation_reason"],
    "additionalProperties": False,
}


class DraftingUnavailable(RuntimeError):
    pass


def drafting_configured() -> bool:
    # in the container, credentials come from the environment (.env -> docker compose)
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client():
    import anthropic

    if not drafting_configured():
        # SDK 1.x doesn't fail at construction without credentials - only at request time,
        # with a plain TypeError - so check up front and say what to do
        raise DraftingUnavailable("reply drafting disabled: set ANTHROPIC_API_KEY in .env")
    return anthropic.Anthropic()


def _format_cases(similar: List[Dict]) -> str:
    parts = []
    for c in similar:
        parts.append(
            f"<past_complaint id=\"{c['complaint_id']}\" product=\"{c['product']}\" "
            f"issue=\"{c.get('issue') or ''}\" company_response=\"{c.get('company_response') or ''}\">\n"
            f"{c['narrative'][:1500]}\n</past_complaint>"
        )
    return "\n".join(parts)


def draft_reply(complaint_text: str, similar: List[Dict], client=None) -> Dict:
    import json

    import anthropic

    client = client or _client()
    user = (f"<similar_past_complaints>\n{_format_cases(similar)}\n</similar_past_complaints>\n\n"
            f"<new_complaint>\n{complaint_text}\n</new_complaint>")
    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # re-run on Anthropic's recommended model if a classifier declines
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": DRAFT_SCHEMA}},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError as e:
        raise DraftingUnavailable("reply drafting disabled: invalid API key") from e
    except anthropic.RateLimitError as e:
        raise DraftingUnavailable("reply drafting rate limited - retry later") from e
    except anthropic.APIConnectionError as e:
        raise DraftingUnavailable("can't reach the Claude API") from e

    if response.stop_reason == "refusal":
        raise DraftingUnavailable("the model declined to draft this reply")
    text = next(b.text for b in response.content if b.type == "text")
    draft = json.loads(text)
    # only keep citations to cases we actually supplied
    supplied = {c["complaint_id"] for c in similar}
    draft["cited_complaint_ids"] = [i for i in draft["cited_complaint_ids"] if i in supplied]
    draft["model"] = response.model
    draft["usage"] = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return draft
