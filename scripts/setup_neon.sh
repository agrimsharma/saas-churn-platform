#!/usr/bin/env bash
# One-time Neon setup for the free demo: asks for the connection string (hidden input, never
# stored in shell history), copies the 40k-complaint vector index into Neon, and saves the
# string as the Modal secret `churn-neon` that deploy/modal_app.py reads.
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL="${MODAL:-$(cd .. && pwd)/.hf-venv/bin/modal}"

read -rsp "Paste your Neon connection string (input hidden): " URL; echo
case "$URL" in postgres://*|postgresql://*) ;; *) echo "That doesn't look like a postgres:// URL"; exit 1 ;; esac
case "$URL" in *sslmode=*) ;; *) URL="$URL$([[ $URL == *\?* ]] && echo '&' || echo '?')sslmode=require" ;; esac

echo "1/2 copying the complaint index to Neon..."
TARGET_DATABASE_URL="$URL" ./scripts/copy_index_to_postgres.sh
echo "2/2 saving it as the Modal secret churn-neon..."
"$MODAL" secret create churn-neon DATABASE_URL="$URL" --force >/dev/null
echo "Done."
