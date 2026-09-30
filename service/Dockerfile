FROM python:3.11-slim

# xgboost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt requirements-nlp.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# DistilBERT complaints classifier: +~1 GB, so opt-in (WITH_NLP=true docker compose build)
ARG WITH_NLP=false
RUN if [ "$WITH_NLP" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-nlp.txt; \
    fi

COPY service/ service/
COPY scripts/ scripts/
# demo reports (drift summary, backtest, example drafts, n8n snapshot) + the classifier's label map
COPY reports/ reports/
COPY models/metrics.json models/metrics.json
# Hugging Face models (embeddings) cache on the mounted models/ volume, not in the image
ENV HF_HOME=/app/models/hf_cache

# docker compose mounts data/, models/ and reports/ over these; Kubernetes uses the baked copies
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
