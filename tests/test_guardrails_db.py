"""The guardrails' Postgres event store - the path the public demo actually uses (limits counted
across restarts, an advisory lock against races). Needs Postgres at TEST_DATABASE_URL (default:
the compose db on localhost:5433, churn_test database); skipped when it isn't reachable."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from service import guardrails, rag
from tests.test_rag import TEST_DB


@pytest.fixture()
def db(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    admin_url = TEST_DB.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = 'churn_test'").fetchone():
                admin.execute("CREATE DATABASE churn_test")
    except psycopg.OperationalError:
        pytest.skip("no Postgres reachable at TEST_DATABASE_URL")
    monkeypatch.setattr(rag, "DATABASE_URL", TEST_DB)
    with guardrails._db() as conn:
        conn.execute("TRUNCATE agent_events")


def test_limits_settle_and_status_in_postgres(db):
    lim = guardrails.Limits(questions_per_day=3, usd_per_day=1.0, usd_per_month=5.0, usd_per_question=0.05,
                            visitor_per_hour=2, visitor_per_day=10)
    e1 = guardrails.reserve("v1", "q1", require_db=True, limits=lim)
    e2 = guardrails.reserve("v1", "q2", require_db=True, limits=lim)
    with pytest.raises(guardrails.LimitReached, match="last hour"):
        guardrails.reserve("v1", "q3", require_db=True, limits=lim)
    guardrails.settle(e1, "answered", 0.02)
    guardrails.settle(e2, "failed", 0)  # failed questions stop counting
    s = guardrails.status("v1", limits=lim)
    assert s["remaining_for_you"] == 1 and s["spent_today_usd"] == pytest.approx(0.02)
    guardrails.reserve("v2", "q4", require_db=True, limits=lim)
    guardrails.reserve("v3", "q5", require_db=True, limits=lim)
    with pytest.raises(guardrails.LimitReached, match="question budget"):
        guardrails.reserve("v4", "q6", require_db=True, limits=lim)


def test_visitor_salt_is_stored_once_and_reused(db, monkeypatch):
    monkeypatch.setattr(guardrails, "_salt", None)
    first = guardrails.visitor_id("203.0.113.7")
    monkeypatch.setattr(guardrails, "_salt", None)  # e.g. after a restart
    assert guardrails.visitor_id("203.0.113.7") == first != guardrails.visitor_id("203.0.113.8")


def test_concurrent_reservations_never_overshoot(db):
    lim = guardrails.Limits(questions_per_day=5, usd_per_question=0.05)

    def try_reserve(i):
        try:
            guardrails.reserve(f"v{i}", "q", require_db=True, limits=lim)
            return True
        except guardrails.LimitReached:
            return False
    with ThreadPoolExecutor(max_workers=10) as pool:
        assert sum(pool.map(try_reserve, range(20))) == 5
