"""Expanding-window validation and held-out test splits."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class ForecastFold:
    """One immutable expanding-window forecast evaluation fold."""

    name: str
    forecast_origin: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    target_dates: tuple[pd.Timestamp, ...]
    role: Literal["validation", "test"]


def make_evaluation_splits(
    dates: pd.DatetimeIndex,
    horizon_days: int = 7,
    validation_folds: int = 8,
    test_days: int = 28,
) -> tuple[list[ForecastFold], list[ForecastFold]]:
    """Reserve the final test period and build expanding seven-day folds."""
    if (horizon_days, validation_folds, test_days) != (7, 8, 28):
        raise ValueError(
            "horizon_days=7, validation_folds=8, and test_days=28 are fixed project invariants"
        )

    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().sort_values()
    if normalized.has_duplicates:
        raise ValueError("dates must be unique")
    if normalized.empty:
        raise ValueError("dates must not be empty")
    expected = pd.date_range(normalized.min(), normalized.max(), freq="D")
    if not normalized.equals(expected):
        raise ValueError("dates must be contiguous daily dates")

    test_folds = test_days // horizon_days
    target_fold_count = validation_folds + test_folds
    required_days = target_fold_count * horizon_days + 1
    if len(normalized) < required_days:
        raise ValueError(f"at least {required_days} dates are required")

    first_target_position = len(normalized) - target_fold_count * horizon_days
    final_test_start = len(normalized) - test_days
    final_test_train_end = pd.Timestamp(normalized[final_test_start - 1])
    folds: list[ForecastFold] = []
    for index in range(target_fold_count):
        target_start = first_target_position + index * horizon_days
        targets = tuple(
            pd.Timestamp(value) for value in normalized[target_start : target_start + 7]
        )
        role: Literal["validation", "test"] = (
            "validation" if index < validation_folds else "test"
        )
        role_index = index + 1 if role == "validation" else index - validation_folds + 1
        origin = pd.Timestamp(normalized[target_start - 1])
        folds.append(
            ForecastFold(
                name=f"{role}_{role_index:02d}",
                forecast_origin=origin,
                train_start=pd.Timestamp(normalized.min()),
                train_end=origin if role == "validation" else final_test_train_end,
                target_dates=targets,
                role=role,
            )
        )

    return folds[:validation_folds], folds[validation_folds:]
