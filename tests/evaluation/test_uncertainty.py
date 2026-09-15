import numpy as np
import pandas as pd
import pytest

from fnb_forecast.evaluation.uncertainty import ResidualIntervalCalibrator


def make_residuals(role: str = "validation") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin": pd.date_range("2025-01-01", periods=4),
            "target_date": pd.date_range("2025-01-02", periods=4),
            "item_id": ["coffee", "coffee", "tea", "tea"],
            "category": ["drink", "drink", "food", "food"],
            "horizon": [1, 2, 1, 2],
            "actual": [10.0, 12.0, 8.0, 9.0],
            "yhat": [8.0, 14.0, 10.0, 8.0],
            "role": role,
        }
    )


def test_interval_uses_only_validation_residuals() -> None:
    calibrator = ResidualIntervalCalibrator(coverage=0.80)
    calibrator.fit(make_residuals(role="validation"))

    with pytest.raises(ValueError, match="validation"):
        calibrator.fit(make_residuals(role="test"))


def test_interval_clamps_lower_and_upper_to_nonnegative() -> None:
    calibrator = ResidualIntervalCalibrator(coverage=0.80)
    calibrator.fit(make_residuals())
    predictions = pd.DataFrame(
        {
            "item_id": ["coffee"],
            "category": ["drink"],
            "horizon": [1],
            "yhat": [0.1],
        }
    )

    result = calibrator.apply(predictions)

    assert result["lower"].ge(0).all()
    assert result["upper"].ge(result["lower"]).all()
    assert np.isfinite(result[["lower", "upper"]].to_numpy()).all()


def test_interval_falls_back_to_horizon_quantiles_for_small_groups() -> None:
    calibrator = ResidualIntervalCalibrator(coverage=0.80, min_group_samples=20)
    calibrator.fit(make_residuals())

    assert all(key[0] == "__horizon__" for key in calibrator.state["quantiles"])
