"""Classify custom messages with the trained model."""
import argparse
from pathlib import Path

import joblib

from src.preprocessing import download_nltk_resources, preprocess_text

MODEL_DIR = Path("models")


def load_model(model_key: str = "tfidf") -> dict:
    return joblib.load(MODEL_DIR / f"{model_key}.joblib")


def predict_message(text: str, artifacts: dict, threshold: float = 0.5) -> dict:
    """Return the predicted label and class probabilities for one raw message.

    `threshold` is the P(spam) cutoff. Raise it (e.g. 0.9) to favour precision.
    Note: Naive Bayes probabilities tend to be overconfident (they are not
    calibrated), so treat them as scores rather than literal likelihoods.
    """
    cleaned = preprocess_text(text)  # must match the training-time preprocessing
    features = artifacts["vectorizer"].transform([cleaned])
    p_spam = float(artifacts["model"].predict_proba(features)[0, 1])
    return {
        "message": text,
        "label": "spam" if p_spam >= threshold else "ham",
        "spam_probability": round(p_spam, 4),
        "ham_probability": round(1 - p_spam, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SMS spam prediction")
    parser.add_argument("messages", nargs="*", help="messages to classify")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    download_nltk_resources()
    artifacts = load_model()
    samples = args.messages or [
        "Congratulations! You won $1,000! Call 0900-555-0100 now to claim your prize.",
        "Hey, are we still meeting for lunch at 1pm tomorrow?",
        "URGENT! Your account has been selected for a FREE gift. Text WIN to 80082.",
        "Can you pick up some milk on your way home?",
    ]
    for msg in samples:
        r = predict_message(msg, artifacts, args.threshold)
        print(f"[{r['label'].upper():>4}] P(spam)={r['spam_probability']:.3f} | {msg}")


if __name__ == "__main__":
    main()