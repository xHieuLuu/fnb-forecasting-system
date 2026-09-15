import numpy as np
import pandas as pd

from fnb_forecast.evaluation.metrics import (
    forecast_bias,
    mae,
    paired_bootstrap_wape_difference,
    rmsse,
    wape,
)


def test_metrics_match_hand_calculation() -> None:
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([2.0, 8.0, 25.0])

    assert wape(actual, predicted) == 9 / 30
    assert mae(actual, predicted) == 3
    assert forecast_bias(actual, predicted) == 5 / 30
    assert np.isclose(
        rmsse(
            actual,
            predicted,
            np.array([1, 2, 3, 4, 5, 6, 7, 8]),
            season_length=7,
        ),
        np.sqrt(11 / 49),
    )


def test_wape_zero_denominator_returns_nan() -> None:
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


def test_paired_bootstrap_samples_complete_item_fold_blocks() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["coffee", "coffee", "tea", "tea"],
            "fold_name": ["fold_1", "fold_1", "fold_2", "fold_2"],
            "actual": [10.0, 10.0, 20.0, 20.0],
            "model_a": [8.0, 8.0, 16.0, 16.0],
            "model_b": [9.0, 9.0, 20.0, 20.0],
        }
    )

    result = paired_bootstrap_wape_difference(
        predictions,
        "model_a",
        "model_b",
        n_bootstrap=2_000,
        seed=17,
    )

    assert set(result) == {"mean_difference", "lower_95", "upper_95"}
    assert np.isclose(result["mean_difference"], 1 / 6)
    assert np.isclose(result["lower_95"], 0.1)
    assert np.isclose(result["upper_95"], 0.2)


def test_paired_bootstrap_point_difference_is_independent_of_sampling() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["coffee", "tea"],
            "fold_name": ["fold_1", "fold_1"],
            "actual": [10.0, 20.0],
            "model_a": [8.0, 16.0],
            "model_b": [9.0, 20.0],
        }
    )

    short_run = paired_bootstrap_wape_difference(
        predictions, "model_a", "model_b", n_bootstrap=20, seed=1
    )
    long_run = paired_bootstrap_wape_difference(
        predictions, "model_a", "model_b", n_bootstrap=2_000, seed=999
    )

    assert np.isclose(short_run["mean_difference"], 1 / 6)
    assert long_run["mean_difference"] == short_run["mean_difference"]


def test_paired_bootstrap_discards_zero_demand_resamples() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["zero_demand", "coffee"],
            "fold_name": ["fold_1", "fold_1"],
            "actual": [0.0, 10.0],
            "model_a": [0.0, 8.0],
            "model_b": [0.0, 9.0],
        }
    )

    result = paired_bootstrap_wape_difference(
        predictions,
        "model_a",
        "model_b",
        n_bootstrap=2_000,
        seed=17,
    )

    assert np.isclose(result["mean_difference"], 0.1)
    assert np.isfinite(result["lower_95"])
    assert np.isclose(result["lower_95"], 0.1)
    assert np.isclose(result["upper_95"], 0.1)


def test_paired_bootstrap_returns_nan_when_no_resample_has_demand() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["coffee", "tea"],
            "fold_name": ["fold_1", "fold_1"],
            "actual": [0.0, 0.0],
            "model_a": [0.0, 0.0],
            "model_b": [0.0, 0.0],
        }
    )

    result = paired_bootstrap_wape_difference(
        predictions, "model_a", "model_b", n_bootstrap=20, seed=17
    )

    assert np.isnan(result["mean_difference"])
    assert np.isnan(result["lower_95"])
    assert np.isnan(result["upper_95"])
