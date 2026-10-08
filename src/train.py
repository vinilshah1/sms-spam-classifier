"""Train and evaluate Multinomial Naive Bayes on BoW and TF-IDF features."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import clean_data, download_dataset, load_raw, split_data
from src.preprocessing import add_clean_text, download_nltk_resources

FIG_DIR = Path("reports/figures")
MODEL_DIR = Path("models")

FEATURE_SETS = {
    "bow": ("Bag-of-Words", CountVectorizer),
    "tfidf": ("TF-IDF", TfidfVectorizer),
}


def compute_metrics(y_true, y_pred) -> dict:
    """Metrics for the positive class (spam = 1)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    # sklearn layout: rows = actual, columns = predicted -> [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    download_nltk_resources()

    df = add_clean_text(clean_data(load_raw(download_dataset())))
    train_df, test_df = split_data(df)  # 80/20, random_state=42, stratified
    y_train, y_test = train_df["label_num"], test_df["label_num"]
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    rows = {}
    for key, (name, vectorizer_cls) in FEATURE_SETS.items():
        vectorizer = vectorizer_cls()
        X_train = vectorizer.fit_transform(train_df["clean_text"])  # fit on TRAIN only
        X_test = vectorizer.transform(test_df["clean_text"])
        print(f"\n[{name}] vocabulary size: {len(vectorizer.vocabulary_)}")

        model = MultinomialNB()  # alpha=1.0 (Laplace smoothing)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"[{name}] Confusion counts: TN={tn} FP={fp} FN={fn} TP={tp}")

        rows[name] = compute_metrics(y_test, y_pred)
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))
        plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix: {name}",
                              FIG_DIR / f"confusion_matrix_{key}.png")
        joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_DIR / f"{key}.joblib")

    results = pd.DataFrame(rows).T.round(4)
    print("\n=== Model comparison (positive class = spam) ===")
    print(results)
    results.to_csv("reports/metrics.csv")
    Path("reports/metrics.json").write_text(json.dumps(results.to_dict("index"), indent=2))


if __name__ == "__main__":
    main()