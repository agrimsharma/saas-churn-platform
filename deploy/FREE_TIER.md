# Going live for free (Modal + Neon, no card anywhere)

**Cost: ₹0.** Modal's Starter plan gives $30/month of compute credit with no card on file, and the
demo scales to zero when idle. Neon's free Postgres has no card either. The public demo never
calls Claude (drafting shows real examples instead). The first visit after an idle period takes
~30 s while the container starts.

## 1. Classifier → Hugging Face Hub (done)

`python3 scripts/publish_hf.py` published https://huggingface.co/agrim-sharma/cfpb-complaints-distilbert

## 2. Database → Neon (~5 min)

1. neon.tech → sign up (Google/GitHub) → create a project (region: Europe).
2. Copy the connection string (Dashboard → Connect; keep `?sslmode=require`).
3. Copy the 40k-complaint vector index into it (~10 s, no re-embedding; needs `docker compose up -d db`):
   ```bash
   TARGET_DATABASE_URL='postgresql://...neon.tech/neondb?sslmode=require' ./scripts/copy_index_to_postgres.sh
   ```

## 3. App → Modal (~15 min, the first image build is the slow part)

```bash
~/Downloads/Projects/.hf-venv/bin/modal token new      # once, if not done for the doppelganger
modal secret create churn-neon DATABASE_URL='postgresql://...neon.tech/neondb?sslmode=require'
modal deploy deploy/modal_app.py
```
→ `https://<workspace>--churn-platform.modal.run`
