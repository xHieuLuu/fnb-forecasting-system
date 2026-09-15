import pandas as pd
import pytest

from fnb_forecast.evaluation.splits import ForecastFold, make_evaluation_splits


def test_splits_reserve_final_28_days_and_have_seven_day_horizons() -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="D")

    validation, test = make_evaluation_splits(dates)

    assert len(validation) == 8
    assert len(test) == 4
    assert all(len(fold.target_dates) == 7 for fold in validation + test)
    assert max(date for fold in validation for date in fold.target_dates) < min(
        date for fold in test for date in fold.target_dates
    )
    assert max(date for fold in test for date in fold.target_dates) == dates.max()
    final_test_start = dates[-28]
    assert all(fold.train_end < final_test_start for fold in test)


def test_split_origins_precede_contiguous_targets_and_training_expands() -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="D")

    validation, test = make_evaluation_splits(dates)
    folds = validation + test

    assert [fold.train_start for fold in folds] == [dates.min()] * 12
    assert [fold.train_end for fold in validation] == [
        fold.forecast_origin for fold in validation
    ]
    assert [fold.train_end for fold in test] == [dates[-29]] * 4
    assert all(
        fold.target_dates
        == tuple(
            pd.date_range(
                fold.forecast_origin + pd.Timedelta(1, unit="days"), periods=7
            )
        )
        for fold in folds
    )
    assert [fold.forecast_origin for fold in folds] == sorted(
        fold.forecast_origin for fold in folds
    )
    assert [fold.name for fold in validation] == [
        f"validation_{index:02d}" for index in range(1, 9)
    ]
    assert [fold.name for fold in test] == [f"test_{index:02d}" for index in range(1, 5)]


@pytest.mark.parametrize(
    ("horizon_days", "validation_folds", "test_days"),
    [(14, 8, 28), (7, 7, 28), (7, 9, 28), (7, 8, 21), (7, 8, 35)],
)
def test_split_project_invariants_are_fixed(
    horizon_days: int, validation_folds: int, test_days: int
) -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="D")

    with pytest.raises(ValueError):
        make_evaluation_splits(dates, horizon_days, validation_folds, test_days)


def test_forecast_fold_is_immutable() -> None:
    fold = ForecastFold(
        name="validation_01",
        forecast_origin=pd.Timestamp("2024-01-07"),
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-07"),
        target_dates=tuple(pd.date_range("2024-01-08", periods=7)),
        role="validation",
    )

    with pytest.raises(AttributeError):
        fold.name = "changed"  # type: ignore[misc]
