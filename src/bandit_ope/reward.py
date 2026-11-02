"""Reward models used by the direct-method and doubly robust estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

N_ARMS = 80
AFFINITY_COLUMNS = [f"user-item_affinity_{arm}" for arm in range(N_ARMS)]
USER_COLUMNS = [f"user_feature_{index}" for index in range(4)]
CATEGORICAL_COLUMNS = ["item_id", "position", *USER_COLUMNS]


def _selected_affinity(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame[AFFINITY_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    actions = frame["item_id"].to_numpy(dtype=np.int64, copy=False)
    return np.log1p(matrix[np.arange(len(frame)), actions])


def _model_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[["item_id", "position", *USER_COLUMNS]].copy()
    features["action_affinity"] = _selected_affinity(frame)
    return features


@dataclass
class ItemPositionRate:
    """Smoothed item-by-position historical CTR baseline."""

    prior_strength: float = 100.0

    def fit(self, frame: pd.DataFrame) -> ItemPositionRate:
        self.global_rate_ = float(frame["click"].mean())
        grouped = frame.groupby(["position", "item_id"])["click"].agg(["sum", "count"])
        self.rates_ = {
            (int(position), int(item)): float(
                (row["sum"] + self.prior_strength * self.global_rate_)
                / (row["count"] + self.prior_strength)
            )
            for (position, item), row in grouped.iterrows()
        }
        return self

    def predict_logged(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                self.rates_.get((int(pos), int(item)), self.global_rate_)
                for pos, item in zip(frame["position"], frame["item_id"], strict=True)
            ],
            dtype=np.float64,
        )

    def predict_target_value(self, contexts: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                np.mean(
                    [self.rates_.get((int(pos), arm), self.global_rate_) for arm in range(N_ARMS)]
                )
                for pos in contexts["position"]
            ],
            dtype=np.float64,
        )

    def get_params(self) -> dict[str, Any]:
        return {"name": "item_position_rate", "prior_strength": self.prior_strength}


class ContextualLogisticReward:
    """Sparse one-hot logistic reward model with user affinity as a numeric feature."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        encoder = OneHotEncoder(
            handle_unknown="ignore", min_frequency=2, sparse_output=True, dtype=np.float64
        )
        preprocessor = ColumnTransformer(
            [("categorical", encoder, CATEGORICAL_COLUMNS)],
            remainder="passthrough",
            sparse_threshold=1.0,
        )
        self.pipeline_ = Pipeline(
            [
                ("features", preprocessor),
                (
                    "reward",
                    LogisticRegression(
                        C=c,
                        solver="liblinear",
                        max_iter=150,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> ContextualLogisticReward:
        features = _model_features(frame)
        self.pipeline_.fit(features, frame["click"].astype(int))
        return self

    def predict_logged(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(_model_features(frame))[:, 1]

    def predict_target_value(self, contexts: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """Average predicted click probability over all 80 target-policy arms."""
        values = np.empty(len(contexts), dtype=np.float64)
        for start in range(0, len(contexts), batch_size):
            batch = contexts.iloc[start : start + batch_size]
            affinity = batch[AFFINITY_COLUMNS].to_numpy(dtype=np.float32, copy=False)
            n_rows = len(batch)
            candidates = (
                batch[["position", *USER_COLUMNS]]
                .loc[batch.index.repeat(N_ARMS)]
                .reset_index(drop=True)
            )
            candidates["item_id"] = np.tile(np.arange(N_ARMS), n_rows)
            candidates["action_affinity"] = np.log1p(affinity.reshape(-1))
            predictions = self.pipeline_.predict_proba(candidates)[:, 1]
            values[start : start + n_rows] = predictions.reshape(n_rows, N_ARMS).mean(axis=1)
        return values

    def get_params(self) -> dict[str, Any]:
        return {
            "name": "contextual_logistic",
            "C": self.c,
            "categorical_features": CATEGORICAL_COLUMNS,
            "numeric_feature": "log1p(affinity between current user and selected item)",
            "max_iter": 150,
        }


def validation_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "mean_prediction": float(np.mean(clipped)),
    }
