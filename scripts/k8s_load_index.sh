#!/usr/bin/env bash
# Load the embedded complaint index (40k rows + HNSW index) into the cluster's Postgres - no
# re-embedding.
#   ./scripts/k8s_load_index.sh <namespace> <release>              # from the local compose db
#   ./scripts/k8s_load_index.sh <namespace> <release> --from-url    # cloud-to-cloud from another
#       Postgres (e.g. Neon): asks for its URL (hidden) and runs pg_dump INSIDE the cluster, so a
#       slow home upload isn't in the path
set -euo pipefail
NS="${1:-churn}"; RELEASE="${2:-churn}"; MODE="${3:-}"
cd "$(dirname "$0")/.."
DB="pod/$RELEASE-db-0"
kubectl -n "$NS" wait --for=condition=ready "$DB" --timeout=300s >/dev/null
kubectl -n "$NS" exec -i "$DB" -- psql -U churn -d churn -q -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector"
DUMP="--table=complaints --no-owner --no-privileges --clean --if-exists"

if [ "$MODE" = "--from-url" ]; then
  read -rsp "Source Postgres URL (input hidden): " SRC; echo
  # A throwaway pod with a NEWER pg_dump (pg_dump refuses servers newer than itself, e.g. Neon on
  # Postgres 18): it dumps from the source and loads into the cluster db. The source URL travels
  # over stdin, never as an argument; the cluster password comes straight from its Secret.
  OVERRIDES=$(cat <<JSON
{"spec": {"containers": [{"name": "copy", "image": "postgres:18", "stdin": true, "stdinOnce": true,
  "command": ["sh", "-c", "read -r SRC; pg_dump \"\$SRC\" $DUMP | psql -h $RELEASE-db -U churn -d churn -q -v ON_ERROR_STOP=1"],
  "env": [{"name": "PGPASSWORD", "valueFrom": {"secretKeyRef": {"name": "$RELEASE-secrets", "key": "POSTGRES_PASSWORD"}}}]}]}}
JSON
)
  printf '%s\n' "$SRC" | kubectl -n "$NS" run index-copy --rm -i --quiet --restart=Never \
    --image=postgres:18 --overrides="$OVERRIDES"
else
  docker compose up -d db >/dev/null
  docker compose exec -T db pg_dump -U churn -d churn $DUMP \
    | kubectl -n "$NS" exec -i "$DB" -- psql -U churn -d churn -q -v ON_ERROR_STOP=1
fi
kubectl -n "$NS" exec "$DB" -- psql -U churn -d churn -tAc "SELECT count(*) || ' complaints loaded' FROM complaints"
