import os
import re
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn_with_all_feedback.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "telco_text_masked_v2.csv")

# v1 masking (scripts/prepare_text.py) only stripped the literal word "churn"/"churning".
# Found a deeper, structural leak: every review ends with a near-identical formulaic verdict
# sentence ("...would recommend this company to others." / "...have no plans to switch
# providers.") that the LLM appended consistently based on the true label - not something a
# real customer review reliably does. Fix: drop the final sentence entirely, keep the actual
# complaint/praise content in the middle, which is the part with genuine (messier) signal.

CHURN_WORD_RE = re.compile(r"\bchurn\w*\b", re.IGNORECASE)


def strip_verdict_sentence(text):
    # split into sentences, drop the last non-empty one (the formulaic verdict)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return ""  # nothing left if it was a one-sentence review
    kept = sentences[:-1]
    cleaned = " ".join(kept)
    cleaned = CHURN_WORD_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    df["CustomerFeedback_masked_v2"] = df["CustomerFeedback"].apply(strip_verdict_sentence)

    empty = (df["CustomerFeedback_masked_v2"].str.len() == 0).sum()
    print(f"Rows with nothing left after stripping (was single-sentence): {empty} / {len(df)}")

    print("\nExample before/after:")
    sample = df.iloc[0]
    print("BEFORE:", sample["CustomerFeedback"])
    print("AFTER: ", sample["CustomerFeedback_masked_v2"])

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} rows to {OUT_PATH}")
