from dataclasses import replace

import pandas as pd
import pytest

from fnb_forecast.contracts import DataBundle
from fnb_forecast.evaluation.backtest import BacktestRunner
from fnb_forecast.evaluation.splits import ForecastFold


class SpyModel:
    model_id = "spy"

    def __init__(self) -> None:
        self.fit_history_end: pd.Timestamp | None = None
        self.predict_history_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        self.fit_history_end = pd.Timestamp(history["date"].max())
        assert known_covariates.empty
        assert "quantity" not in known_covariates

    def predict(
        self, history: pd.DataFrame, future_covariates: pd.DataFrame
    ) -> pd.DataFrame:
        self.predict_history_end = pd.Timestamp(history["date"].max())
        assert "quantity" not in future_covariates
        assert set(future_covariates["source"]) == {"previous_runs_fixture"}
        assert future_covariates["issued_at"].notna().all()
        origin = pd.Timestamp(future_covariates["target_date"].min()) - pd.Timedelta(
            1, unit="D"
        )
        return future_covariates.loc[:, ["target_date", "item_id"]].assign(
            forecast_origin=origin,
            yhat=5.0,
            model_id=self.model_id,
        )

    def save(self, path: object) -> None:
        raise NotImplementedError


class DuplicateSpyModel(SpyModel):
    def predict(
        self, history: pd.DataFrame, future_covariates: pd.DataFrame
    ) -> pd.DataFrame:
        forecast = super().predict(history, future_covariates)
        return pd.concat([forecast, forecast.iloc[[0]]], ignore_index=True)


def make_backtest_bundle(valid_bundle: DataBundle) -> DataBundle:
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    items = ["coffee", "tea"]
    sales = pd.MultiIndex.from_product(
        [items, dates], names=["item_id", "date"]
    ).to_frame(index=False)
    sales["quantity"] = range(1, len(sales) + 1)
    sales["unit_price"] = 30_000.0
    sales["promo_flag"] = pd.Series(False, index=sales.index, dtype="boolean")
    sales["stockout_flag"] = pd.Series(False, index=sales.index, dtype="boolean")
    sales["store_open"] = pd.Series(True, index=sales.index, dtype="boolean")
    calendar = pd.DataFrame(
        {
            "date": dates,
            "weekday": dates.weekday,
            "is_weekend": pd.Series(dates.weekday >= 5, dtype="boolean"),
            "holiday_name": [None] * len(dates),
            "days_to_tet": range(28, 0, -1),
            "days_after_tet": [None] * len(dates),
            "store_open": pd.Series(True, index=range(len(dates)), dtype="boolean"),
        }
    )
    target_dates = dates[14:]
    weather = pd.DataFrame(
        {
            "issued_at": pd.Timestamp("2025-01-10", tz="Asia/Ho_Chi_Minh"),
            "target_date": target_dates,
            "temperature": 30.0,
            "rain": 0.0,
            "humidity": 70.0,
            "source": "previous_runs_fixture",
        }
    )
    return replace(
        valid_bundle,
        daily_sales=sales,
        item_master=pd.DataFrame(
            {
                "item_id": items,
                "item_name": ["Coffee", "Tea"],
                "category": ["beverage", "beverage"],
                "launch_date": [dates.min(), dates.min()],
                "active": pd.Series(True, index=range(len(items)), dtype="boolean"),
            }
        ),
        calendar=calendar,
        weather=weather,
    )


def make_folds() -> list[ForecastFold]:
    shared_train_end = pd.Timestamp("2025-01-14")
    return [
        ForecastFold(
            name="validation_01",
            forecast_origin=shared_train_end,
            train_start=pd.Timestamp("2025-01-01"),
            train_end=shared_train_end,
            target_dates=tuple(pd.date_range("2025-01-15", periods=7)),
            role="validation",
        ),
        ForecastFold(
            name="test_02",
            forecast_origin=pd.Timestamp("2025-01-21"),
            train_start=pd.Timestamp("2025-01-01"),
            train_end=shared_train_end,
            target_dates=tuple(pd.date_range("2025-01-22", periods=7)),
            role="test",
        ),
    ]


def test_runner_uses_train_end_and_never_exposes_future_targets(
    valid_bundle: DataBundle,
) -> None:
    bundle = make_backtest_bundle(valid_bundle)
    folds = make_folds()
    models: list[SpyModel] = []

    def model_factory() -> SpyModel:
        model = SpyModel()
        models.append(model)
        return model

    result = BacktestRunner().run(model_factory, bundle, folds)

    assert len(models) == 2
    assert models[0] is not models[1]
    assert [model.fit_history_end for model in models] == [
        fold.train_end for fold in folds
    ]
    assert [model.predict_history_end for model in models] == [
        fold.forecast_origin for fold in folds
    ]
    assert len(result) == 7 * 2 * 2
    assert not result.duplicated(
        ["forecast_origin", "target_date", "item_id"]
    ).any()
    assert set(result["fold_name"]) == {"validation_01", "test_02"}
    assert set(result["role"]) == {"validation", "test"}
    assert result["dataset_name"].unique().tolist() == [bundle.provenance.name]
    expected_actual = bundle.daily_sales.rename(
        columns={"date": "target_date", "quantity": "actual"}
    ).loc[:, ["target_date", "item_id", "actual"]]
    joined = result.merge(
        expected_actual,
        on=["target_date", "item_id"],
        suffixes=("", "_expected"),
        validate="many_to_one",
    )
    assert joined["actual"].equals(joined["actual_expected"])


def test_runner_rejects_duplicate_forecast_keys(valid_bundle: DataBundle) -> None:
    bundle = make_backtest_bundle(valid_bundle)

    with pytest.raises(ValueError, match="unique"):
        BacktestRunner().run(DuplicateSpyModel, bundle, make_folds()[:1])


def test_runner_rejects_missing_target_rows_for_required_item(
    valid_bundle: DataBundle,
) -> None:
    bundle = make_backtest_bundle(valid_bundle)
    target_dates = pd.DatetimeIndex(make_folds()[0].target_dates)
    missing_tea_targets = (bundle.daily_sales["item_id"] == "tea") & (
        bundle.daily_sales["date"].isin(target_dates)
    )
    bundle = replace(
        bundle,
        daily_sales=bundle.daily_sales.loc[~missing_tea_targets].copy(),
    )

    with pytest.raises(ValueError, match="every item and target date"):
        BacktestRunner().run(SpyModel, bundle, make_folds()[:1])
