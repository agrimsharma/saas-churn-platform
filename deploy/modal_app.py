"""
Free public demo on Modal (Starter plan: $30/month of credit, no card on file).

One container runs the API (localhost only) and the Streamlit dashboard (public). It starts on
the first visit, stays warm SCALEDOWN_S seconds after the last one, then scales to zero. The
image does the slow work at build time: Telco data, both language models and a pre-trained churn
model are baked in, so a cold start only loads them. PUBLIC_DEMO disables everything that changes
state, and Claude reply drafting; the Claude agent runs under a daily question budget
(AGENT_DAILY_LIMIT, counted in Neon so it survives restarts).

One-time setup (see deploy/FREE_TIER.md):
  python scripts/publish_hf.py                                   # classifier -> HF Hub (free)
  modal secret create churn-neon DATABASE_URL='postgresql://...' # Neon (pgvector) connection
  ./scripts/setup_agent_secret.sh                                # Anthropic key for the agent
Deploy:
  modal deploy deploy/modal_app.py      -> https://<workspace>--churn-platform.modal.run
"""
import os
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = "/root/app"
SCALEDOWN_S = 300
AGENT_DAILY_LIMIT = "20"             # agent questions per UTC day, across all visitors
AGENT_MODEL = "claude-opus-5-5"
COMPLAINTS_MODEL = os.environ.get("COMPLAINTS_MODEL_REPO", "agrim-sharma/cfpb-complaints-distilbert")
TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
REPORTS = ["example_drafts.json", "demo_actions.jsonl", "retail_backtest.json",
           "rag_retrieval_eval.json", "drift_summary.json"]

app = modal.App("churn-platform")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "curl")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "requirements-api.txt"))
    .pip_install_from_requirements(str(ROOT / "requirements-nlp.txt"))
    .pip_install("streamlit>=1.50.0", "requests")
    .env({
        "PUBLIC_DEMO": "true",
        "HF_HOME": "/root/hf",
        "COMPLAINTS_MODEL": COMPLAINTS_MODEL,
        "CHURN_DATA_PATH": f"{APP}/data/raw/Telco-Customer-Churn.csv",
        "CHURN_API_URL": "http://127.0.0.1:8000",
        "REPORTS_DIR": f"{APP}/reports",
        "AGENT_DAILY_LIMIT": AGENT_DAILY_LIMIT,
        "CLAUDE_AGENT_MODEL": AGENT_MODEL,
    })
    .run_commands(f"mkdir -p {APP}/data/raw && curl -fsSL -o {APP}/data/raw/Telco-Customer-Churn.csv {TELCO_URL}")
    # pre-download the embedding model and the fine-tuned classifier into the image
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); "
        f"from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; "
        f"AutoTokenizer.from_pretrained('{COMPLAINTS_MODEL}'); M.from_pretrained('{COMPLAINTS_MODEL}')\""
    )
    .add_local_dir(str(ROOT / "service"), f"{APP}/service", copy=True, ignore=["__pycache__", "Dockerfile"])
    .add_local_dir(str(ROOT / "dashboard"), f"{APP}/dashboard", copy=True, ignore=["Dockerfile"])
    .add_local_file(str(ROOT / "models" / "metrics.json"), f"{APP}/models/metrics.json", copy=True)
    .run_commands(f"mkdir -p {APP}/reports")
)
for name in REPORTS:
    image = image.add_local_file(str(ROOT / "reports" / name), f"{APP}/reports/{name}", copy=True)
# train + save the serving model at build time (~5 s), so containers load it instead
image = image.workdir(APP).run_commands("python -c 'from service import churn_model; churn_model.save(churn_model.train())'")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("churn-neon"),       # DATABASE_URL: complaint index + agent budget
             modal.Secret.from_name("churn-anthropic")],  # ANTHROPIC_API_KEY for the agent
    cpu=1.0,
    memory=4096,
    scaledown_window=SCALEDOWN_S,
    max_containers=1,  # Streamlit sessions are per-container; also caps spend
    timeout=3600,
)
@modal.concurrent(max_inputs=50)
@modal.web_server(8501, startup_timeout=240, label="churn-platform")
def dashboard():
    import subprocess
    import time
    import urllib.request

    subprocess.Popen(["uvicorn", "service.app:app", "--host", "127.0.0.1", "--port", "8000"], cwd=APP)
    for _ in range(120):  # the dashboard needs the API on its first render
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    subprocess.Popen(["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501",
                      "--server.headless=true", "--browser.gatherUsageStats=false"], cwd=APP)
