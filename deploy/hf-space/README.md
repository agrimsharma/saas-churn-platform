---
title: SaaS Churn Platform
emoji: 📉
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Churn scoring, drift monitoring, complaint RAG
---

Free public demo of [saas-churn-platform](https://github.com/agrimsharma/saas-churn-platform):
calibrated XGBoost churn scoring with SHAP risk drivers, PSI drift monitoring, a retraining
backtest on real time-stamped data, and complaint triage (DistilBERT classifier + pgvector
retrieval over 40k CFPB complaints).

The n8n workflows and live Claude reply drafting run in the full deployment only; this Space
never makes billed API calls.
