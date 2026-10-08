"""Run the entire pipeline end to end:  uv run python -m src"""
import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk, never open windows

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

from src import eda, error_analysis, train  # noqa: E402
from src.predict import load_model, predict_message  # noqa: E402
from src.preprocessing import download_nltk_resources  # noqa: E402

LOG_PATH = Path("reports/run_log.txt")
DEMO_MESSAGES = [
    "Congratulations! You won $1,000! Call 0900-555-0100 now to claim your prize.",
    "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "URGENT! Your account has been selected for a FREE gift. Text WIN to 80082.",
    "Can you pick up some milk on your way home?",
]


class _Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def run_step(title: str, fn) -> None:
    print(f"\n{'#' * 80}\n# {title}\n{'#' * 80}")
    start = time.perf_counter()
    fn()
    print(f"\n[done in {time.perf_counter() - start:.1f}s]")


def demo_predictions() -> None:
    download_nltk_resources()
    artifacts = load_model("tfidf")
    for msg in DEMO_MESSAGES:
        r = predict_message(msg, artifacts)
        print(f"[{r['label'].upper():>4}] P(spam)={r['spam_probability']:.3f} | {msg}")


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log, redirect_stdout(_Tee(sys.stdout, log)):
        run_step("1/4  Exploratory data analysis", eda.main)
        run_step("2/4  Training and evaluation", train.main)
        run_step("3/4  Error analysis", error_analysis.main)
        run_step("4/4  Demo predictions", demo_predictions)
    print(f"\nAll done. Figures: reports/figures/ | Metrics: reports/metrics.csv | Log: {LOG_PATH}")


if __name__ == "__main__":
    main()