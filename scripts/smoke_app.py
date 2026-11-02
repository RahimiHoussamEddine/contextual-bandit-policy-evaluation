"""Exercise the rendered Streamlit app and a clipping control interaction."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    if app.exception:
        raise RuntimeError(f"Streamlit app raised: {app.exception[0].message}")
    if not app.title or app.title[0].value != "Contextual Bandit Policy Evaluation":
        raise AssertionError("Dashboard title did not render")
    if len(app.slider) != 1 or len(app.get("plotly_chart")) < 3:
        raise AssertionError("Expected one clipping slider and three evidence charts")
    initial_snips = next(metric.value for metric in app.metric if metric.label == "SNIPS")
    app.slider[0].set_value(5.0).run()
    if app.exception:
        raise RuntimeError(f"Dashboard failed after changing the cap: {app.exception[0].message}")
    changed_snips = next(metric.value for metric in app.metric if metric.label == "SNIPS")
    if initial_snips == changed_snips:
        raise AssertionError("Changing the weight cap did not update the SNIPS estimate")
    print("Streamlit render passed; changing the weight cap updated the SNIPS estimate.")


if __name__ == "__main__":
    main()
