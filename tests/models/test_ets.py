import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd
import pytest

from fnb_forecast.contracts import DataBundle
from fnb_forecast.evaluation.backtest import BacktestRunner
from fnb_forecast.evaluation.splits import ForecastFold
from fnb_forecast.models import ETSModel, SeasonalNaiveModel


def test_ets_uses_fitted_statsforecast_2_api_and_clamps_negative_predictions(
    make_model_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    history, future = make_model_frames(values=list(range(28, 0, -1)), horizon=7)
    model = ETSModel(season_length=7)

    model.fit(history, future.iloc[0:0])
    result = model.predict(history, future)

    assert result["target_date"].tolist() == future["target_date"].tolist()
    assert result["model_id"].unique().tolist() == ["auto_ets_7"]
    assert (result["yhat"] >= 0.0).all()
    assert result["yhat"].eq(0.0).any()


def test_ets_saves_versioned_metadata_and_fitted_joblib(
    tmp_path: Path,
    make_model_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    history, future = make_model_frames(values=list(range(1, 29)), horizon=7)
    model = ETSModel(season_length=7)
    model.fit(history, future.iloc[0:0])

    model.save(tmp_path)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["model_id"] == "auto_ets_7"
    assert metadata["season_length"] == 7
    with pytest.warns(DeprecationWarning, match="Setting the shape"):
        fitted = joblib.load(tmp_path / "model.joblib")
    assert hasattr(fitted, "fitted_")
    assert "AutoETS" in fitted.predict(h=7).columns


def test_ets_backtest_joins_requested_dates_after_each_items_last_non_null_target(
    baseline_backtest_case: tuple[DataBundle, list[ForecastFold]],
) -> None:
    bundle, folds = baseline_backtest_case
    sales = bundle.daily_sales.copy()
    day_number = sales.groupby("item_id").cumcount() + 1
    item_offset = sales["item_id"].map({"coffee": 0, "tea": 100})
    sales["quantity"] = day_number + item_offset
    tea_trailing_nulls = (sales["item_id"] == "tea") & sales["date"].between(
        "2025-01-22", "2025-01-28"
    )
    sales.loc[tea_trailing_nulls, "quantity"] = None
    delayed_fold = replace(folds[1], train_end=pd.Timestamp("2025-01-28"))

    result = BacktestRunner().run(
        ETSModel,
        replace(bundle, daily_sales=sales),
        [delayed_fold],
    )

    coffee = result.loc[result["item_id"] == "coffee"]
    tea = result.loc[result["item_id"] == "tea"]
    assert coffee["target_date"].tolist() == list(pd.date_range("2025-02-05", periods=7))
    assert coffee["yhat"].tolist() == pytest.approx(list(range(36, 43)))
    assert tea["target_date"].tolist() == list(pd.date_range("2025-02-05", periods=7))
    assert tea["yhat"].tolist() == pytest.approx(list(range(136, 143)))


@pytest.mark.parametrize("model_factory", [SeasonalNaiveModel, ETSModel])
def test_statistical_baselines_run_two_items_and_two_folds_through_backtester(
    model_factory: Callable[[], SeasonalNaiveModel | ETSModel],
    baseline_backtest_case: tuple[DataBundle, list[ForecastFold]],
) -> None:
    bundle, folds = baseline_backtest_case

    result = BacktestRunner().run(model_factory, bundle, folds)

    required = {"forecast_origin", "target_date", "item_id", "yhat", "model_id"}
    assert required.issubset(result.columns)
    assert len(result) == 2 * 2 * 7
    assert not result.duplicated(["forecast_origin", "target_date", "item_id"]).any()
    assert (result["yhat"] >= 0.0).all()
    for fold in folds:
        fold_result = result.loc[result["fold_name"] == fold.name]
        for item_id in ["coffee", "tea"]:
            item_dates = fold_result.loc[
                fold_result["item_id"] == item_id, "target_date"
            ].tolist()
            assert item_dates == list(fold.target_dates)
