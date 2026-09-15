"""Forecast accuracy metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return weighted absolute percentage error."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_values).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual_values - predicted_values).sum() / denominator)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return mean absolute error."""
    return float(
        np.mean(
            np.abs(
                np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
            )
        )
    )


def rmsse(
    actual: np.ndarray,
    predicted: np.ndarray,
    training: np.ndarray,
    season_length: int = 7,
) -> float:
    """Return root mean squared scaled error."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    training_values = np.asarray(training, dtype=float)
    scale = np.mean(
        (training_values[season_length:] - training_values[:-season_length]) ** 2
    )
    if scale == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual_values - predicted_values) ** 2) / scale))


def forecast_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return signed forecast error relative to total actual demand."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    denominator = actual_values.sum()
    if denominator == 0:
        return float("nan")
    return float((predicted_values - actual_values).sum() / denominator)


def paired_bootstrap_wape_difference(
    predictions: pd.DataFrame,
    model_a_column: str,
    model_b_column: str,
    *,
    n_bootstrap: int = 2_000,
    seed: int = 20260813,
) -> dict[str, float]:
    """Compare two models by resampling paired item/fold forecast blocks."""
    block_rows: list[tuple[float, float, float]] = []
    for _, block in predictions.groupby(["item_id", "fold_name"], sort=True):
        actual = block["actual"].to_numpy(dtype=float)
        denominator = float(np.abs(actual).sum())
        error_a = float(np.abs(actual - block[model_a_column].to_numpy(dtype=float)).sum())
        error_b = float(np.abs(actual - block[model_b_column].to_numpy(dtype=float)).sum())
        block_rows.append((denominator, error_a, error_b))

    blocks = np.asarray(block_rows, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(n_bootstrap, len(blocks)))
    totals = blocks[sampled].sum(axis=1)
    valid_totals = totals[totals[:, 0] > 0]
    differences = valid_totals[:, 1] / valid_totals[:, 0] - (
        valid_totals[:, 2] / valid_totals[:, 0]
    )
    differences = differences[np.isfinite(differences)]
    if differences.size:
        lower, upper = np.quantile(differences, [0.025, 0.975])
    else:
        lower = upper = float("nan")
    point_difference = wape(
        predictions["actual"].to_numpy(dtype=float),
        predictions[model_a_column].to_numpy(dtype=float),
    ) - wape(
        predictions["actual"].to_numpy(dtype=float),
        predictions[model_b_column].to_numpy(dtype=float),
    )
    return {
        "mean_difference": point_difference,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }
