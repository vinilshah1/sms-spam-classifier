"""Data loading, cleaning and splitting for the UCI SMS Spam Collection."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_PATH = Path("data/raw/SMSSpamCollection")
LABEL_MAP = {"ham": 0, "spam": 1}

RANDOM_STATE = 42  # fixed seed -> reproducible split
TEST_SIZE = 0.2    # 80/20 split


def download_dataset(dest: Path = RAW_PATH) -> Path:
    """Download and extract the dataset once. If the URL fails, download the
    zip manually from the UCI repository and place the file at `dest`."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            zf.extract("SMSSpamCollection", dest.parent)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """The UCI file is tab-separated with no header: <label>\t<message>.
    If you get a UnicodeDecodeError, retry with encoding='latin-1'.
    (The Kaggle 'spam.csv' variant needs latin-1 and has columns v1, v2 plus
    junk 'Unnamed: 2-4' columns: rename v1->label, v2->message and drop the rest.)
    """
    return pd.read_csv(path, sep="\t", header=None, names=["label", "message"])


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Handle nulls, normalise text, drop duplicates and encode labels (ham=0, spam=1)."""
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["label", "message"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[(df["message"] != "") & df["label"].isin(LABEL_MAP)]

    # Exact duplicates are common in this dataset. Keeping them risks the same
    # message landing in both train and test (data leakage -> inflated scores).
    if drop_duplicates:
        df = df.drop_duplicates(subset="message")

    df["label_num"] = df["label"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split so both sets keep the ham/spam ratio.
    Returns DataFrames (not arrays) so we keep the raw text for error analysis."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_num"]
    )
