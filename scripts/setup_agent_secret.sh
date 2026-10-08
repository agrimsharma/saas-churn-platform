#!/usr/bin/env bash
# One-time: saves an Anthropic API key as the Modal secret `churn-anthropic`, which the free demo's
# Claude agent reads (hidden input, never stored in shell history). Use a key from a separate
# Anthropic workspace with a monthly spend limit, so the public demo can never cost more than that.
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL="${MODAL:-$(cd .. && pwd)/.hf-venv/bin/modal}"

read -rsp "Paste the Anthropic API key for the public demo (input hidden): " KEY; echo
case "$KEY" in sk-ant-*) ;; *) echo "That doesn't look like an Anthropic API key (sk-ant-...)"; exit 1 ;; esac
"$MODAL" secret create churn-anthropic ANTHROPIC_API_KEY="$KEY" --force >/dev/null
echo "Saved. Deploy with: $MODAL deploy deploy/modal_app.py"
