# Going live for free (Hugging Face Space + Neon, no card anywhere)

**Cost: ₹0, permanently.** The public demo never calls Claude (drafting shows real examples instead),
and neither service has a card on file. The Space sleeps after ~48 h idle and wakes on the next visit.

## 1. Database → Neon (~5 min)

1. neon.tech → sign up (Google/GitHub) → create a project (region: Europe).
2. Copy the connection string (Dashboard → Connect; keep `?sslmode=require`).
3. Copy the 40k-complaint vector index into it (~10 s, no re-embedding; needs `docker compose up -d db`):
   ```bash
   TARGET_DATABASE_URL='postgresql://...neon.tech/neondb?sslmode=require' ./scripts/copy_index_to_postgres.sh
   ```

## 2. App → Hugging Face Space (~15 min)

1. Log in once (a **Write** token from huggingface.co → Settings → Access Tokens):
   ```bash
   python3 -m pip install --user -U huggingface_hub && hf auth login
   ```
2. Publish the classifier (public model repo) and the Space; the Neon URL becomes a Space secret:
   ```bash
   NEON_DATABASE_URL='postgresql://...neon.tech/neondb?sslmode=require' python3 scripts/publish_hf.py
   ```
3. After the build (~10 min): `https://<you>-churn-platform.hf.space` → the dashboard.
