"""Embed CFPB complaint narratives into pgvector for similar-ticket retrieval.

Source: the CFPB Consumer Complaint Database with narratives (CC0 mirror on Hugging Face,
BEE-spoke-data/consumer-finance-complaints, one parquet shard). The current official bulk
export no longer includes the narrative text, so it can't be used for retrieval.

Holds out a random evaluation set (never inserted) and scores retrieval on it:
  precision@k (same product), precision@k (same product AND issue), vs a TF-IDF baseline.

Run inside the API container (needs the NLP extras + the db service):
  docker compose exec api python scripts/build_complaint_index.py --n 40000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from service import rag  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SOURCE = os.path.join(ROOT, "data", "raw", "cfpb", "cfpb-narratives-00006.parquet")
REPORT = os.path.join(ROOT, "reports", "rag_retrieval_eval.json")
COLS = {
    "Complaint ID": "complaint_id", "Date received": "date_received", "Product": "product",
    "Sub-product": "sub_product", "Issue": "issue", "Sub-issue": "sub_issue", "Company": "company",
    "Company response to consumer": "company_response", "Timely response?": "timely_response",
    "State": "state", "Consumer complaint narrative": "narrative",
}
# CFPB renamed products over the years; merge the renames so "same product" is meaningful
PRODUCT_ALIASES = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Credit card",
    "Payday loan, title loan, or personal loan": "Personal / payday loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal / payday loan",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Bank account or service": "Checking or savings account",
}


def load(n_total: int, since: str, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(SOURCE, columns=list(COLS))
    df = df.rename(columns=COLS).dropna(subset=["narrative"])
    df = df[pd.to_datetime(df["date_received"]) >= since]
    df = df[df["narrative"].str.len() >= 100]  # drop one-liners
    df["product"] = df["product"].replace(PRODUCT_ALIASES)
    df["timely_response"] = df["timely_response"].eq("Yes")
    df["date_received"] = pd.to_datetime(df["date_received"]).dt.date
    df["complaint_id"] = df["complaint_id"].astype("int64")
    return df.drop_duplicates("complaint_id").sample(n=min(n_total, len(df)), random_state=seed).reset_index(drop=True)


def precision_at_k(queries: pd.DataFrame, retrieved: list, k: int):
    same_product, same_issue = [], []
    for (_, q), hits in zip(queries.iterrows(), retrieved):
        hits = hits[:k]
        same_product.append(np.mean([h["product"] == q["product"] for h in hits]))
        same_issue.append(np.mean([h["product"] == q["product"] and h["issue"] == q["issue"] for h in hits]))
    return float(np.mean(same_product)), float(np.mean(same_issue))


def tfidf_baseline(corpus: pd.DataFrame, queries: pd.DataFrame, k: int):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=100_000, stop_words="english")
    C = vec.fit_transform(corpus["narrative"])
    Q = vec.transform(queries["narrative"])
    top = np.argsort(-(Q @ C.T).toarray(), axis=1)[:, :k]
    rows = corpus[["product", "issue"]].to_dict("records")
    return [[rows[j] for j in r] for r in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000, help="complaints to index")
    ap.add_argument("--eval", type=int, default=1000, help="held-out queries (not indexed)")
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    df = load(args.n + args.eval, args.since, seed=42)
    held_out, corpus = df.iloc[:args.eval], df.iloc[args.eval:]
    print(f"{len(corpus):,} to index, {len(held_out):,} held out ({time.time() - t0:.0f}s)")

    conn = rag.connect()
    rag.init_schema(conn)
    conn.execute("TRUNCATE complaints")
    batch = 512
    for start in range(0, len(corpus), batch):
        chunk = corpus.iloc[start:start + batch]
        emb = rag.embed(chunk["narrative"].tolist())
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO complaints ({', '.join(rag.COLUMNS)}, embedding) "
                f"VALUES ({', '.join(['%s'] * (len(rag.COLUMNS) + 1))})",
                [tuple(r[c] for c in rag.COLUMNS) + (e,) for r, e in zip(chunk.to_dict("records"), emb)],
            )
        if (start // batch) % 10 == 0:
            print(f"  indexed {min(start + batch, len(corpus)):,} ({time.time() - t0:.0f}s)")
    conn.execute("ANALYZE complaints")

    # --- retrieval evaluation on held-out complaints -------------------------------------
    q_emb = rag.embed(held_out["narrative"].tolist())
    t_search = time.time()
    retrieved = [rag.search(conn, e, k=args.k) for e in q_emb]
    search_ms = (time.time() - t_search) / len(held_out) * 1000
    emb_prod, emb_issue = precision_at_k(held_out, retrieved, args.k)
    tf_prod, tf_issue = precision_at_k(held_out, tfidf_baseline(corpus, held_out, args.k), args.k)

    # chance level: probability a random indexed complaint shares the query's product / issue
    p = corpus.groupby("product").size() / len(corpus)
    pi = corpus.groupby(["product", "issue"]).size() / len(corpus)
    chance_prod = float(held_out["product"].map(p).fillna(0).mean())
    chance_issue = float(pd.Series(list(zip(held_out["product"], held_out["issue"]))).map(pi).fillna(0).mean())

    report = {
        "indexed": int(len(corpus)), "held_out_queries": int(len(held_out)), "k": args.k,
        "since": args.since, "embedding_model": rag.EMBED_MODEL,
        "precision_at_k": {
            "embeddings_same_product": round(emb_prod, 4), "embeddings_same_product_and_issue": round(emb_issue, 4),
            "tfidf_same_product": round(tf_prod, 4), "tfidf_same_product_and_issue": round(tf_issue, 4),
            "chance_same_product": round(chance_prod, 4), "chance_same_product_and_issue": round(chance_issue, 4),
        },
        "mean_search_ms": round(search_ms, 2),
        "build_seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
