#!/usr/bin/env bash
# Copy the embedded complaint index (40k rows + HNSW index) from the local compose db into the
# cluster's Postgres - no re-embedding.   ./scripts/k8s_load_index.sh <namespace> <release>
set -euo pipefail
NS="${1:-churn}"; RELEASE="${2:-churn}"
cd "$(dirname "$0")/.."
docker compose up -d db >/dev/null
kubectl -n "$NS" wait --for=condition=ready "pod/$RELEASE-db-0" --timeout=300s >/dev/null
kubectl -n "$NS" exec -i "$RELEASE-db-0" -- psql -U churn -d churn -q -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector"
docker compose exec -T db pg_dump -U churn -d churn --table=complaints --no-owner --no-privileges --clean --if-exists \
  | kubectl -n "$NS" exec -i "$RELEASE-db-0" -- psql -U churn -d churn -q -v ON_ERROR_STOP=1
kubectl -n "$NS" exec "$RELEASE-db-0" -- psql -U churn -d churn -tAc "SELECT count(*) || ' complaints loaded' FROM complaints"
