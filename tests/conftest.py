"""Shared fixtures: one freshly trained model in a temp dir, used by every API test."""
import pytest
from fastapi.testclient import TestClient

from service import app as app_module
from service import churn_model


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    mp = pytest.MonkeyPatch()
    mp.setattr(churn_model, "MODEL_DIR", tmp)
    mp.setattr(churn_model, "PIPELINE_PATH", tmp / "pipeline.joblib")
    mp.setattr(churn_model, "META_PATH", tmp / "meta.json")
    mp.setattr(app_module, "ACTIONS_LOG_PATH", tmp / "actions.jsonl")
    mp.setattr(app_module, "API_KEY", None)
    app_module._state["model"] = None
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


