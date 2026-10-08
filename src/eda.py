"""Exploratory data analysis: class balance and message-length statistics."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import clean_data, download_dataset, load_raw

FIG_DIR = Path("reports/figures")
PALETTE = {"ham": "#4C72B0", "spam": "#C44E52"}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    raw = load_raw(download_dataset())
    print(f"Raw shape: {raw.shape}")
    print(f"Null values:\n{raw.isna().sum()}\n")
    print(f"Duplicate messages: {raw.duplicated(subset='message').sum()}")

    df = clean_data(raw)
    print(f"Shape after cleaning/deduplication: {df.shape}\n")

    # ---- Class distribution ----
    counts = df["label"].value_counts()
    print("Class counts:\n", counts)
    print("\nClass percentages:\n", (df["label"].value_counts(normalize=True) * 100).round(2))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="label", hue="label", order=["ham", "spam"],
                  palette=PALETTE, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, padding=2)
    ax.set(title="Class Distribution", xlabel="Class", ylabel="Number of messages")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.show()

    # ---- Basic text statistics ----
    df["char_len"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()
    df["digit_count"] = df["message"].str.count(r"\d")
    print("\nText statistics by class:")
    print(df.groupby("label")[["char_len", "word_count", "digit_count"]].describe().T)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col, title in zip(
        axes,
        ["char_len", "word_count", "digit_count"],
        ["Characters per message", "Words per message", "Digits per message"],
    ):
        sns.histplot(data=df, x=col, hue="label", palette=PALETTE, bins=40,
                     stat="density", common_norm=False, element="step", ax=ax)
        ax.set(title=title, xlabel=col.replace("_", " "))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "text_statistics.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
