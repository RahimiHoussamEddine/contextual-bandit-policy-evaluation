import numpy as np
import pandas as pd
import pytest

from bandit_ope.data import _validate


def sample_frame(rows: int = 10_000) -> pd.DataFrame:
    days = np.repeat(pd.date_range("2019-11-24", periods=5).strftime("%Y-%m-%d"), rows // 5)
    return pd.DataFrame(
        {
            "source_row_id": np.arange(rows),
            "timestamp": pd.to_datetime(days, utc=True),
            "item_id": np.arange(rows) % 80,
            "position": np.arange(rows) % 3 + 1,
            "click": np.arange(rows) % 2,
            "propensity_score": np.full(rows, 1 / 80),
        }
    )


def test_source_validation_accepts_well_formed_five_day_sample() -> None:
    stats = _validate(sample_frame(), "random")
    assert stats["rows"] == 10_000
    assert len(stats["days"]) == 5
    assert stats["source_row_id_min"] == 0
    assert stats["source_row_id_max"] == 9_999


def test_source_validation_rejects_duplicate_row_ids() -> None:
    frame = sample_frame()
    frame.loc[10, "source_row_id"] = frame.loc[9, "source_row_id"]
    with pytest.raises(ValueError, match="duplicated or out of order"):
        _validate(frame, "bts")


def test_source_validation_rejects_out_of_catalog_arm() -> None:
    frame = sample_frame()
    frame.loc[0, "item_id"] = 80
    with pytest.raises(ValueError, match="80-arm catalog"):
        _validate(frame, "bts")
