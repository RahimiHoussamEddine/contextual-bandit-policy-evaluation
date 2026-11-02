"""Transparent estimators and diagnostics for a uniform item-level target policy."""

from __future__ import annotations

from typing import Any

import numpy as np


def importance_weights(propensity: np.ndarray, target_probability: float) -> np.ndarray:
    behavior = np.asarray(propensity, dtype=np.float64)
    if behavior.ndim != 1 or np.any(~np.isfinite(behavior)) or np.any(behavior <= 0):
        raise ValueError("Behavior propensities must be a finite positive vector")
    if not 0 < target_probability <= 1:
        raise ValueError("Target action probability must lie in (0, 1]")
    return target_probability / behavior


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    weight = np.asarray(weights, dtype=np.float64)
    if len(weight) == 0 or np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("Importance weights must be a nonempty finite nonnegative vector")
    total = float(weight.sum())
    squared = float(np.square(weight).sum())
    ess = total * total / squared if squared else 0.0
    return {
        "mean": float(weight.mean()),
        "median": float(np.median(weight)),
        "p95": float(np.quantile(weight, 0.95)),
        "p99": float(np.quantile(weight, 0.99)),
        "max": float(weight.max()),
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / len(weight),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    estimate = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "estimate": estimate,
        "naive_iid_se": standard_error,
        "naive_iid_95ci_low": max(0.0, estimate - 1.96 * standard_error),
        "naive_iid_95ci_high": min(1.0, estimate + 1.96 * standard_error),
        "rows": int(len(values)),
    }


def estimate_ips(
    reward: np.ndarray, weights: np.ndarray, cap: float | None = None
) -> dict[str, Any]:
    clipped = np.minimum(weights, cap) if cap is not None else weights
    return {
        **_summary(clipped * reward),
        "weights": weight_diagnostics(clipped),
        "cap": cap,
        "fraction_clipped": float(np.mean(weights > cap)) if cap is not None else 0.0,
    }


def estimate_snips(
    reward: np.ndarray, weights: np.ndarray, cap: float | None = None
) -> dict[str, Any]:
    clipped = np.minimum(weights, cap) if cap is not None else weights
    denominator = float(clipped.sum())
    if denominator <= 0:
        raise ValueError("SNIPS has zero total importance weight")
    estimate = float(np.sum(clipped * reward) / denominator)
    influence = clipped * (reward - estimate) / clipped.mean()
    standard_error = float(influence.std(ddof=1) / np.sqrt(len(influence)))
    return {
        "estimate": estimate,
        "naive_iid_se": standard_error,
        "naive_iid_95ci_low": max(0.0, estimate - 1.96 * standard_error),
        "naive_iid_95ci_high": min(1.0, estimate + 1.96 * standard_error),
        "rows": int(len(reward)),
        "weights": weight_diagnostics(clipped),
        "cap": cap,
        "fraction_clipped": float(np.mean(weights > cap)) if cap is not None else 0.0,
    }


def estimate_dr(
    direct_values: np.ndarray,
    observed_reward: np.ndarray,
    observed_prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    contributions = direct_values + weights * (observed_reward - observed_prediction)
    return {**_summary(contributions), "weights": weight_diagnostics(weights)}


def estimate_dm(direct_values: np.ndarray) -> dict[str, Any]:
    return _summary(direct_values)
