#!/usr/bin/env bash
# API on localhost only (not exposed); the dashboard is the public entry point.
# DATABASE_URL (Neon, a Space secret) and COMPLAINTS_MODEL (Hub repo id, a Space variable)
# enable the complaint tab; without them the rest of the dashboard still works.
set -euo pipefail
uvicorn service.app:app --host 127.0.0.1 --port 8000 &
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 1
done
exec streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=7860 \
  --server.headless=true --browser.gatherUsageStats=false
