import numpy as np
import pytest

from bandit_ope.estimators import (
    estimate_dr,
    estimate_ips,
    estimate_snips,
    importance_weights,
    weight_diagnostics,
)


def test_importance_weights_and_effective_sample_size() -> None:
    weights = importance_weights(np.array([0.25, 0.5]), target_probability=0.125)
    np.testing.assert_allclose(weights, [0.5, 0.25])
    assert weight_diagnostics(weights)["effective_sample_size"] == pytest.approx(1.8)


def test_ips_and_snips_match_hand_calculation() -> None:
    reward = np.array([1.0, 0.0])
    weights = np.array([1.0, 2.0])
    assert estimate_ips(reward, weights)["estimate"] == pytest.approx(0.5)
    assert estimate_snips(reward, weights)["estimate"] == pytest.approx(1 / 3)
    assert estimate_ips(reward, weights, cap=0.5)["estimate"] == pytest.approx(0.25)


def test_doubly_robust_formula() -> None:
    result = estimate_dr(
        direct_values=np.array([0.3, 0.4]),
        observed_reward=np.array([1.0, 0.0]),
        observed_prediction=np.array([0.2, 0.3]),
        weights=np.array([2.0, 1.0]),
    )
    assert result["estimate"] == pytest.approx(1.0)


@pytest.mark.parametrize("propensity", [np.array([0.0]), np.array([np.nan]), np.array([-0.1])])
def test_invalid_propensities_are_rejected(propensity: np.ndarray) -> None:
    with pytest.raises(ValueError):
        importance_weights(propensity, target_probability=0.1)
