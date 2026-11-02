"""Select a reward model using only the predeclared validation dates."""

from __future__ import annotations

import json

import joblib

from bandit_ope.reward import (
    ContextualLogisticReward,
    ItemPositionRate,
    validation_metrics,
)
from bandit_ope.splits import ROOT, load_policy, split_by_day


def train() -> dict[str, object]:
    bts = split_by_day(load_policy("bts"))
    training, validation = bts["train"], bts["validation"]
    candidates: list[tuple[str, object]] = [
        ("item_position_rate", ItemPositionRate().fit(training))
    ]
    for c in (0.1, 1.0, 10.0):
        candidates.append(
            (f"contextual_logistic_c_{c:g}", ContextualLogisticReward(c).fit(training))
        )

    results = []
    for name, model in candidates:
        metrics = validation_metrics(
            validation["click"].to_numpy(dtype=int), model.predict_logged(validation)
        )
        results.append({"name": name, **metrics, "model_parameters": model.get_params()})
    selected = min(results, key=lambda result: (result["log_loss"], result["brier_score"]))
    model_lookup = dict(candidates)
    selected_model = model_lookup[str(selected["name"])]

    model_dir = ROOT / "models"
    report_dir = ROOT / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    joblib.dump(selected_model, model_dir / "reward_model.joblib", compress=3)
    output = {
        "selection_rule": "Lowest validation log loss, with Brier score as a tie-breaker.",
        "target_policy": "Uniform random over 80 items at each position.",
        "train_days": sorted(training["day"].unique().tolist()),
        "validation_days": sorted(validation["day"].unique().tolist()),
        "train_rows": len(training),
        "validation_rows": len(validation),
        "train_click_rate": float(training["click"].mean()),
        "validation_click_rate": float(validation["click"].mean()),
        "selected_model": selected,
        "candidate_results": results,
    }
    path = report_dir / "model_selection.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {selected['name']} on validation log loss {selected['log_loss']:.6f}")
    print(f"Wrote {path.relative_to(ROOT)} and models/reward_model.joblib")
    return output


def main() -> None:
    train()


if __name__ == "__main__":
    main()
