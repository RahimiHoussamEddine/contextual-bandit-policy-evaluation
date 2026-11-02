"""Small read-only API for aggregate evaluation and interactive weight clipping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from bandit_ope.estimators import estimate_ips, estimate_snips

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(
    title="Contextual Bandit Policy Evaluation API",
    version="0.1.0",
    description="Aggregate-only estimates from the local, locked Open Bandit test sample.",
)


def _load_evaluation() -> dict[str, Any]:
    path = ROOT / "reports/evaluation.json"
    if not path.exists():
        raise HTTPException(status_code=409, detail="Run make evaluate before requesting results")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/summary")
def summary() -> dict[str, Any]:
    evaluation = _load_evaluation()
    return {
        "target_policy": evaluation["target_policy"],
        "split": evaluation["split"],
        "random_test_reference": evaluation["random_test_reference"],
        "estimators": evaluation["estimators"],
        "uncertainty_note": evaluation["uncertainty_note"],
    }


@app.get("/estimate")
def estimate(cap: float | None = Query(default=None, gt=0)) -> dict[str, Any]:
    evaluation = _load_evaluation()
    path = ROOT / evaluation["local_contributions_file"]
    if not path.exists():
        raise HTTPException(
            status_code=409, detail="Local row contributions are missing; run make evaluate"
        )
    rows = pd.read_csv(path, compression="gzip")
    weights = rows["importance_weight"].to_numpy()
    rewards = rows["click"].to_numpy()
    reference = float(evaluation["random_test_reference"]["click_rate"])
    ips = estimate_ips(rewards, weights, cap)
    snips = estimate_snips(rewards, weights, cap)
    return {
        "cap": cap,
        "rows": len(rows),
        "random_test_reference": reference,
        "ips": {**ips, "absolute_error_vs_random_reference": abs(ips["estimate"] - reference)},
        "snips": {
            **snips,
            "absolute_error_vs_random_reference": abs(snips["estimate"] - reference),
        },
        "note": "The cap changes estimates and bias; compare it with unclipped IPS and SNIPS.",
    }
