"""Interactive localhost dashboard for off-policy estimator diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bandit_ope.estimators import estimate_ips, estimate_snips

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Bandit OPE Workbench", page_icon="🎯", layout="wide")


@st.cache_data
def load_reports() -> tuple[dict, pd.DataFrame]:
    evaluation = json.loads((ROOT / "reports/evaluation.json").read_text(encoding="utf-8"))
    contributions = pd.read_csv(ROOT / evaluation["local_contributions_file"], compression="gzip")
    return evaluation, contributions


st.title("Contextual Bandit Policy Evaluation")
st.caption(
    "A time-aware comparison of IPS, SNIPS, Direct Method and Doubly Robust estimates "
    "against independent Random-policy logs."
)

report_file = ROOT / "reports/evaluation.json"
if not report_file.exists():
    st.info(
        "Prepare the local sample and evaluation first: run `make fetch`, `make train`, "
        "then `make evaluate`."
    )
    st.stop()

evaluation, contributions = load_reports()
reference = evaluation["random_test_reference"]["click_rate"]
estimators = evaluation["estimators"]
selected_cap = evaluation["clip_selection"]["selected_cap"]

left, middle, right = st.columns(3)
left.metric(
    "Random holdout reference",
    f"{reference:.3%}",
    f"{evaluation['random_test_reference']['rows']:,} rows",
)
middle.metric(
    "Locked BTS test sample",
    f"{evaluation['split']['test_rows_bts']:,} rows",
    ", ".join(evaluation["split"]["test_days"]),
)
right.metric("Validation-selected cap", "None" if selected_cap is None else f"{selected_cap:g}")

st.subheader("Explore importance-weight clipping")
use_clipping = st.checkbox("Clip importance weights", value=True)
max_weight = float(contributions["importance_weight"].max())
slider_max = max(0.5, float(np.ceil(max_weight)))
slider_default = min(slider_max, max(0.5, selected_cap or min(20.0, max_weight)))
clip_cap = st.slider(
    "Maximum importance weight",
    min_value=0.5,
    max_value=slider_max,
    value=slider_default,
    step=0.5,
    disabled=not use_clipping,
)
effective_cap = clip_cap if use_clipping else None
reward = contributions["click"].to_numpy(dtype=float)
weights = contributions["importance_weight"].to_numpy(dtype=float)
ips = estimate_ips(reward, weights, effective_cap)
snips = estimate_snips(reward, weights, effective_cap)
selection, diagnostics = st.columns(2)
selection.metric("IPS", f"{ips['estimate']:.3%}", f"{ips['estimate'] - reference:+.3%} vs Random")
selection.metric(
    "SNIPS", f"{snips['estimate']:.3%}", f"{snips['estimate'] - reference:+.3%} vs Random"
)
diagnostics.metric(
    "Effective sample size",
    f"{ips['weights']['effective_sample_size']:,.0f}",
    f"{ips['weights']['effective_sample_size_fraction']:.1%} of rows",
)
diagnostics.metric("Weights clipped", f"{ips['fraction_clipped']:.1%}", f"raw max {max_weight:.2f}")

plot_left, plot_right = st.columns(2)
with plot_left:
    st.markdown("**OPE estimates on the locked test sample**")
    rows = []
    for name, result in estimators.items():
        rows.append(
            {"Estimator": name.replace("_", " ").title(), "Estimated CTR": result["estimate"]}
        )
    rows.extend(
        [
            {"Estimator": "Interactive clipped IPS", "Estimated CTR": ips["estimate"]},
            {"Estimator": "Interactive clipped SNIPS", "Estimated CTR": snips["estimate"]},
        ]
    )
    chart = px.bar(pd.DataFrame(rows), x="Estimator", y="Estimated CTR", color="Estimator")
    chart.add_hline(
        y=reference, line_dash="dash", line_color="#c44e52", annotation_text="Random holdout"
    )
    chart.update_layout(
        showlegend=False, yaxis_tickformat=".2%", yaxis_title="Estimated click rate"
    )
    st.plotly_chart(chart, width="stretch")

with plot_right:
    st.markdown("**Test-day estimates**")
    daily_rows = []
    for day, result in evaluation["per_day"].items():
        daily_rows.extend(
            [
                {"Day": day, "Series": "Random reference", "CTR": result["random_reference"]},
                {"Day": day, "Series": "IPS", "CTR": result["ips"]["estimate"]},
                {"Day": day, "Series": "SNIPS", "CTR": result["snips"]["estimate"]},
                {"Day": day, "Series": "Doubly robust", "CTR": result["dr"]["estimate"]},
            ]
        )
    daily_chart = px.line(pd.DataFrame(daily_rows), x="Day", y="CTR", color="Series", markers=True)
    daily_chart.update_layout(yaxis_tickformat=".2%", yaxis_title="Estimated click rate")
    st.plotly_chart(daily_chart, width="stretch")

st.subheader("Weight and support diagnostics")
hist = go.Figure()
hist.add_trace(
    go.Histogram(x=np.minimum(weights, effective_cap) if effective_cap else weights, nbinsx=45)
)
hist.update_layout(xaxis_title="Importance weight", yaxis_title="Rows", showlegend=False)
st.plotly_chart(hist, width="stretch")
support = evaluation["support"]
st.write(
    f"Observed {support['observed_position_item_cells']} of "
    f"{support['possible_position_item_cells']} position-item cells in the BTS test sample; "
    f"minimum cell count {support['rows_per_observed_cell_min']}."
)

st.subheader("Estimator comparison")
table = pd.DataFrame(
    [
        {
            "Estimator": name.replace("_", " ").title(),
            "Estimated CTR": result["estimate"],
            "Absolute error vs Random": result["absolute_error_vs_random_reference"],
            "Naive row-IID 95% low": result.get("naive_iid_95ci_low"),
            "Naive row-IID 95% high": result.get("naive_iid_95ci_high"),
        }
        for name, result in estimators.items()
    ]
)
st.dataframe(
    table.style.format({column: "{:.3%}" for column in table.columns if column != "Estimator"}),
    width="stretch",
)
st.warning(evaluation["uncertainty_note"])
with st.expander("Method limits and interpretation"):
    for note in evaluation["method_limitations"]:
        st.markdown(f"- {note}")
    st.markdown(
        "IPS and SNIPS use the logged BTS propensity and the Random policy's 1/80 action "
        "probability. The selected cap is chosen using validation data only. DM and DR use "
        "the reward model selected "
        "on validation log loss; test estimates were not used for model or cap selection."
    )
