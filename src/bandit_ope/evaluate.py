"""Evaluate the uniform random target policy on the locked late-period sample."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bandit_ope.estimators import (
    estimate_dm,
    estimate_dr,
    estimate_ips,
    estimate_snips,
    importance_weights,
)
from bandit_ope.reward import validation_metrics
from bandit_ope.splits import ROOT, load_policy, split_by_day

TARGET_PROBABILITY = 1.0 / 80.0
CLIP_CANDIDATES: tuple[float | None, ...] = (None, 5.0, 10.0, 20.0, 50.0, 100.0)


def _clip_label(cap: float | None) -> str:
    return "unclipped" if cap is None else f"{cap:g}"


def _with_reference(result: dict[str, Any], reference: float) -> dict[str, Any]:
    return {**result, "absolute_error_vs_random_reference": abs(result["estimate"] - reference)}


def _day_estimates(
    day: str,
    mask: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    direct: np.ndarray,
    predicted: np.ndarray,
    selected_cap: float | None,
    random: pd.DataFrame,
) -> dict[str, Any]:
    day_reference = float(random.loc[random["day"].eq(day), "click"].mean())
    result = {
        "rows": int(mask.sum()),
        "random_reference": day_reference,
        "ips": estimate_ips(rewards[mask], weights[mask]),
        "snips": estimate_snips(rewards[mask], weights[mask]),
        "dm": estimate_dm(direct[mask]),
        "dr": estimate_dr(direct[mask], rewards[mask], predicted[mask], weights[mask]),
        "selected_clipped_ips": estimate_ips(rewards[mask], weights[mask], selected_cap),
        "selected_clipped_snips": estimate_snips(rewards[mask], weights[mask], selected_cap),
    }
    for key in ("ips", "snips", "dm", "dr", "selected_clipped_ips", "selected_clipped_snips"):
        result[key] = _with_reference(result[key], day_reference)
    return result


def evaluate() -> dict[str, Any]:
    bts_parts = split_by_day(load_policy("bts"))
    random_parts = split_by_day(load_policy("random"))
    model_path = ROOT / "models/reward_model.joblib"
    selection_path = ROOT / "reports/model_selection.json"
    if not model_path.exists() or not selection_path.exists():
        raise FileNotFoundError("Run `make train` before `make evaluate`")
    model = joblib.load(model_path)
    model_selection = json.loads(selection_path.read_text(encoding="utf-8"))

    validation = bts_parts["validation"]
    random_validation = random_parts["validation"]
    validation_weights = importance_weights(
        validation["propensity_score"].to_numpy(), TARGET_PROBABILITY
    )
    reference_validation = float(random_validation["click"].mean())
    clip_selection = []
    for cap in CLIP_CANDIDATES:
        estimate = estimate_ips(validation["click"].to_numpy(dtype=float), validation_weights, cap)
        clip_selection.append(
            {
                "cap": cap,
                "label": _clip_label(cap),
                "ips_estimate": estimate["estimate"],
                "random_validation_reference": reference_validation,
                "absolute_error": abs(estimate["estimate"] - reference_validation),
                "fraction_clipped": estimate["fraction_clipped"],
                "effective_sample_size": estimate["weights"]["effective_sample_size"],
                "validation_only": True,
            }
        )
    minimum_error = min(row["absolute_error"] for row in clip_selection)
    tied_candidates = [
        row
        for row in clip_selection
        if np.isclose(row["absolute_error"], minimum_error, rtol=0.0, atol=1e-12)
    ]
    selected = min(tied_candidates, key=lambda row: row["cap"] is not None)
    selected_cap = selected["cap"]

    test = bts_parts["test"].reset_index(drop=True)
    random_test = random_parts["test"].reset_index(drop=True)
    random_test_propensities = np.sort(random_test["propensity_score"].unique())
    if len(random_test_propensities) != 1 or not np.isclose(
        random_test_propensities[0], TARGET_PROBABILITY, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "The independent Random logs do not confirm the configured uniform 1/80 propensity"
        )

    rewards = test["click"].to_numpy(dtype=np.float64)
    weights = importance_weights(test["propensity_score"].to_numpy(), TARGET_PROBABILITY)
    observed_prediction = np.clip(model.predict_logged(test), 1e-7, 1 - 1e-7)
    direct_values = np.clip(model.predict_target_value(test), 1e-7, 1 - 1e-7)
    reference = float(random_test["click"].mean())
    estimates = {
        "ips": estimate_ips(rewards, weights),
        "snips": estimate_snips(rewards, weights),
        "dm": estimate_dm(direct_values),
        "dr": estimate_dr(direct_values, rewards, observed_prediction, weights),
        "selected_clipped_ips": estimate_ips(rewards, weights, selected_cap),
        "selected_clipped_snips": estimate_snips(rewards, weights, selected_cap),
    }
    estimates = {name: _with_reference(value, reference) for name, value in estimates.items()}

    support = test.groupby(["position", "item_id"]).size()
    all_cells = 3 * 80
    per_day: dict[str, Any] = {}
    for day in sorted(test["day"].unique()):
        mask = test["day"].eq(day).to_numpy()
        per_day[day] = _day_estimates(
            day,
            mask,
            rewards,
            weights,
            direct_values,
            observed_prediction,
            selected_cap,
            random_test,
        )

    test_reward_metrics = validation_metrics(rewards.astype(int), observed_prediction)
    contribution_path = ROOT / "reports/test_contributions.csv.gz"
    pd.DataFrame(
        {
            "source_row_id": test["source_row_id"].astype("int64"),
            "day": test["day"],
            "click": rewards.astype("int8"),
            "importance_weight": weights,
            "q_logged": observed_prediction,
            "q_target": direct_values,
        }
    ).to_csv(contribution_path, index=False, compression="gzip")

    output = {
        "target_policy": {
            "name": "uniform_random",
            "arms": 80,
            "action_probability_per_position": TARGET_PROBABILITY,
            "reference_confirmation": (
                "All sampled independent Random-policy propensities equal 1/80."
            ),
        },
        "split": {
            "train_days": model_selection["train_days"],
            "validation_days": model_selection["validation_days"],
            "test_days": sorted(test["day"].unique().tolist()),
            "test_rows_bts": len(test),
            "test_rows_random": len(random_test),
            "selection_rule": (
                "Clipping cap chosen on validation-only IPS error to the independent "
                "Random validation reference."
            ),
        },
        "reward_model": {
            **model_selection["selected_model"]["model_parameters"],
            "selected_on": "validation log loss, Brier score tie-breaker",
            "test_observed_action_metrics": test_reward_metrics,
            "test_observed_click_rate": float(rewards.mean()),
        },
        "clip_selection": {
            "selected_cap": selected_cap,
            "random_validation_reference": reference_validation,
            "selection_reason": (
                "Minimum absolute error to the independent Random validation reference; "
                "ties prefer unclipped IPS."
            ),
            "tied_caps": [row["cap"] for row in tied_candidates],
            "candidates": clip_selection,
        },
        "random_test_reference": {
            "click_rate": reference,
            "rows": len(random_test),
            "by_day": {
                day: float(random_test.loc[random_test["day"].eq(day), "click"].mean())
                for day in sorted(random_test["day"].unique())
            },
        },
        "estimators": estimates,
        "weight_diagnostics": estimates["ips"]["weights"],
        "support": {
            "observed_position_item_cells": int(len(support)),
            "possible_position_item_cells": all_cells,
            "missing_cells": int(all_cells - len(support)),
            "rows_per_observed_cell_min": int(support.min()),
            "rows_per_observed_cell_median": float(support.median()),
        },
        "per_day": per_day,
        "uncertainty_note": (
            "The reported Wald intervals use an i.i.d. row approximation and can be too narrow. "
            "The systematic byte-range sample is not a probability sample, and sampled rows may "
            "be dependent; the intervals do not represent uncertainty for the full campaign. "
            "Compare the three test-day estimates."
        ),
        "method_limitations": [
            "The convenience byte-range sample is not probability-sampled and is not "
            "representative by design.",
            "Only six of the seven campaign dates appear in these selected ranges; "
            "2019-11-25 is absent.",
            "Random-policy reference logs are independent but byte windows are not exactly "
            "time-matched to BTS windows.",
            "Each displayed item-position is treated as an item-level decision; slate "
            "interference is not modeled.",
            "The direct model uses logged pre-decision user features and the selected item's "
            "historical affinity.",
            "The Random logs provide a held-out policy benchmark, not a live deployment or "
            "causal guarantee.",
        ],
        "local_contributions_file": str(contribution_path.relative_to(ROOT)),
    }
    report_path = ROOT / "reports/evaluation.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Random reference CTR (test): {reference:.4%} across {len(random_test):,} rows")
    for name, result in estimates.items():
        message = f"{name}: {result['estimate']:.4%}"
        if "weights" in result:
            message += f", ESS {result['weights']['effective_sample_size']:.0f}"
        print(message)
    print(f"Validation-selected clipping cap: {selected_cap}")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    return output


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
