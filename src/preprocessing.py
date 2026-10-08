"""Modular text-preprocessing functions for SMS messages."""
import string
from functools import lru_cache
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Replace punctuation with a space, because SMS text often lacks spaces after
# punctuation ("entry...Text FA to 87121"). Apostrophes are removed separately.
_PUNCT_TO_SPACE = str.maketrans({c: " " for c in string.punctuation})
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def download_nltk_resources() -> None:
    """Fetch the NLTK corpora needed (no-op if already present)."""
    for resource in ("stopwords", "wordnet"):
        nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _stop_words() -> frozenset:
    return frozenset(stopwords.words("english"))


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.replace("'", "").translate(_PUNCT_TO_SPACE)


def tokenize(text: str) -> List[str]:
    """Whitespace tokenisation. It is enough after punctuation removal and
    avoids needing NLTK's 'punkt' download."""
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    stop = _stop_words()
    return [t for t in tokens if t not in stop]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [_stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str, method: str = "stem") -> str:
    """Full pipeline: lowercase -> strip punctuation -> tokenise -> drop
    stop words -> stem/lemmatise. Returns a space-joined string.

    Args:
        method: 'stem', 'lemmatize' or 'none'.
    """
    tokens = tokenize(remove_punctuation(to_lowercase(text)))
    tokens = remove_stopwords(tokens)
    if method == "stem":
        tokens = stem_tokens(tokens)
    elif method == "lemmatize":
        tokens = lemmatize_tokens(tokens)
    elif method != "none":
        raise ValueError("method must be 'stem', 'lemmatize' or 'none'")
    return " ".join(tokens)


def add_clean_text(df: pd.DataFrame, method: str = "stem") -> pd.DataFrame:
    """Add a `clean_text` column computed from `message`."""
    df = df.copy()
    df["clean_text"] = df["message"].apply(preprocess_text, method=method)
    return df
