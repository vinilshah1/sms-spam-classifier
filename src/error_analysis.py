"""Inspect false positives / false negatives of the saved TF-IDF model."""
import textwrap
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import clean_data, download_dataset, load_raw, split_data
from src.preprocessing import add_clean_text, download_nltk_resources

MODEL_DIR = Path("models")


def show_examples(frame: pd.DataFrame, title: str, n: int = 5) -> None:
    print(f"\n{'=' * 80}\n{title}  (showing {min(n, len(frame))} of {len(frame)})\n{'=' * 80}")
    for i, (_, row) in enumerate(frame.head(n).iterrows(), start=1):
        print(f"{i}. P(spam) = {row['spam_prob']:.3f}")
        print(textwrap.indent(textwrap.fill(row["message"], width=76), "   "), "\n")


def main(model_key: str = "tfidf") -> None:
    download_nltk_resources()
    df = add_clean_text(clean_data(load_raw(download_dataset())))
    _, test_df = split_data(df)  # identical split to train.py (same seed)

    artifacts = joblib.load(MODEL_DIR / f"{model_key}.joblib")
    X_test = artifacts["vectorizer"].transform(test_df["clean_text"])

    results = test_df.assign(
        pred=artifacts["model"].predict(X_test),
        spam_prob=artifacts["model"].predict_proba(X_test)[:, 1],
    )

    # False positives: actual ham, predicted spam (most confidently wrong first)
    fp = results[(results.label_num == 0) & (results.pred == 1)].sort_values(
        "spam_prob", ascending=False)
    # False negatives: actual spam, predicted ham (most confidently wrong first)
    fn = results[(results.label_num == 1) & (results.pred == 0)].sort_values("spam_prob")

    show_examples(fp, "FALSE POSITIVES: legitimate messages flagged as spam")
    show_examples(fn, "FALSE NEGATIVES: spam that slipped through")

    # If the model makes fewer than 5 false positives (a good outcome!), show the
    # ham messages that came closest to being flagged.
    if len(fp) < 5:
        near = results[results.label_num == 0].sort_values("spam_prob", ascending=False)
        near = near[near.pred == 0]
        show_examples(near, "NEAR MISSES: ham with the highest spam probability")


if __name__ == "__main__":
    main()