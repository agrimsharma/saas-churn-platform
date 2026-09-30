# Hugging Face Space (Docker SDK, free CPU tier): API (internal) + Streamlit dashboard (public).
# scripts/publish_hf.py stages this folder + service/, dashboard/, reports and pushes it.
# HF runs containers as uid 1000 and routes traffic to port 7860.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
COPY requirements-api.txt requirements-nlp.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt -r requirements-nlp.txt "streamlit>=1.50.0" requests

USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 HF_HOME=/home/user/.cache/huggingface \
    PUBLIC_DEMO=true CHURN_DATA_PATH=/home/user/app/data/raw/Telco-Customer-Churn.csv \
    CHURN_API_URL=http://127.0.0.1:8000 REPORTS_DIR=/home/user/app/reports
WORKDIR /home/user/app
# the public IBM copy of the Telco dataset (same columns the model uses)
RUN mkdir -p data/raw models && curl -fsSL -o data/raw/Telco-Customer-Churn.csv \
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

COPY --chown=user service/ service/
COPY --chown=user dashboard/ dashboard/
COPY --chown=user reports/ reports/
COPY --chown=user models/metrics.json models/metrics.json
COPY --chown=user start.sh .
EXPOSE 7860
CMD ["bash", "start.sh"]
