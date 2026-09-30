import os
import re
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn_with_all_feedback.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "telco_text_masked.csv")

# The literal word "churn"/"churning" is a label-leakage artifact: this feedback was generated
# by GPT conditioned on the TRUE churn status, so it sometimes echoes it directly ("I have
# decided to churn", "no plans to churn"). Masking only this specific word (not genuine
# sentiment words like "switch"/"cancel"/"leave", which are legitimate signal a real customer
# might actually write) removes the shortcut without gutting the real signal.
CHURN_WORD_RE = re.compile(r"\bchurn\w*\b", re.IGNORECASE)


def mask_churn_word(text):
    cleaned = CHURN_WORD_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)              # collapse double spaces left behind
    cleaned = re.sub(r"\s+([.,])", r"\1", cleaned)       # fix " ." -> "."
    return cleaned.strip()


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    before = df["CustomerFeedback"].str.contains("churn", case=False, na=False).sum()
    df["CustomerFeedback_masked"] = df["CustomerFeedback"].apply(mask_churn_word)
    after = df["CustomerFeedback_masked"].str.contains("churn", case=False, na=False).sum()

    print(f"Rows mentioning 'churn' before masking: {before}")
    print(f"Rows mentioning 'churn' after masking:  {after}")
    print("\nExample before/after:")
    sample = df[df["CustomerFeedback"].str.contains("churn", case=False, na=False)].iloc[0]
    print("BEFORE:", sample["CustomerFeedback"])
    print("AFTER: ", sample["CustomerFeedback_masked"])

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} rows to {OUT_PATH}")
