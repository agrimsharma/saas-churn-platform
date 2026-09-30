#!/usr/bin/env bash
# Copy the embedded complaint index from the local compose db to another Postgres+pgvector
# (e.g. Neon's free tier) - no re-embedding. Vectors, metadata and the HNSW index come across.
#   TARGET_DATABASE_URL='postgresql://user:pass@host/db?sslmode=require' ./scripts/copy_index_to_postgres.sh
set -euo pipefail
: "${TARGET_DATABASE_URL:?set TARGET_DATABASE_URL (e.g. your Neon connection string)}"
cd "$(dirname "$0")/.."

# pgvector's extension must exist before the table that uses its type
docker compose exec -T -e TARGET="$TARGET_DATABASE_URL" db sh -c 'psql "$TARGET" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector"'
docker compose exec -T -e TARGET="$TARGET_DATABASE_URL" db sh -c \
  'pg_dump -U churn -d churn --table=complaints --no-owner --no-privileges --clean --if-exists \
   | psql "$TARGET" -v ON_ERROR_STOP=1 --quiet'
docker compose exec -T -e TARGET="$TARGET_DATABASE_URL" db sh -c \
  'psql "$TARGET" -tAc "SELECT count(*) || '"'"' complaints copied'"'"' FROM complaints"'
