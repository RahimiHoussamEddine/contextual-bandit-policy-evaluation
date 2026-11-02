"""Predeclared chronological split for the bounded source-row sample."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DAYS = {"2019-11-24", "2019-11-26"}
VALIDATION_DAYS = {"2019-11-27"}
TEST_DAYS = {"2019-11-28", "2019-11-29", "2019-11-30"}


def load_policy(policy: str) -> pd.DataFrame:
    """Load one local row sample and add its UTC calendar day."""
    path = ROOT / "data/raw" / f"{policy}_all_sample.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `make fetch` first")
    frame = pd.read_csv(path, compression="gzip")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["day"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    return frame.sort_values("source_row_id", kind="stable").reset_index(drop=True)


def split_by_day(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply fixed date sets without shuffling or consulting outcomes."""
    groups = {
        "train": TRAIN_DAYS,
        "validation": VALIDATION_DAYS,
        "test": TEST_DAYS,
    }
    result = {name: frame.loc[frame["day"].isin(days)].copy() for name, days in groups.items()}
    if any(part.empty for part in result.values()):
        available = sorted(frame["day"].unique().tolist())
        missing = {name: sorted(days - set(available)) for name, days in groups.items()}
        raise ValueError(f"Chronological split is missing expected dates: {missing}")
    if not (max(TRAIN_DAYS) < min(VALIDATION_DAYS) and max(VALIDATION_DAYS) < min(TEST_DAYS)):
        raise ValueError("Chronological split dates overlap or are out of order")
    return result
