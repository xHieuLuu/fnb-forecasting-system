from collections.abc import Callable
from dataclasses import replace

import pandas as pd
import pytest

from fnb_forecast.contracts import DataBundle
from fnb_forecast.evaluation.splits import ForecastFold


@pytest.fixture
def make_model_frames() -> Callable[..., tuple[pd.DataFrame, pd.DataFrame]]:
    def make_frames(
        *,
        values: list[float | int | None],
        horizon: int = 7,
        item_id: str = "coffee",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
        history = pd.DataFrame(
            {
                "date": dates,
                "item_id": item_id,
                "quantity": values,
                "unit_price": 30_000.0,
                "promo_flag": False,
                "stockout_flag": False,
                "store_open": True,
            }
        )
        future = pd.DataFrame(
            {
                "target_date": pd.date_range(
                    dates[-1] + pd.Timedelta(1, unit="D"), periods=horizon
                ),
                "item_id": item_id,
                "unit_price": 30_000.0,
                "promo_flag": False,
            }
        )
        return history, future

    return make_frames


@pytest.fixture
def baseline_backtest_case(
    valid_bundle: DataBundle,
) -> tuple[DataBundle, list[ForecastFold]]:
    dates = pd.date_range("2025-01-01", periods=42, freq="D")
    items = ["coffee", "tea"]
    sales = pd.MultiIndex.from_product(
        [items, dates], names=["item_id", "date"]
    ).to_frame(index=False)
    day_number = sales.groupby("item_id").cumcount()
    item_offset = sales["item_id"].map({"coffee": 1, "tea": 8})
    sales["quantity"] = (day_number % 7) + item_offset
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
            "days_to_tet": range(42, 0, -1),
            "days_after_tet": [None] * len(dates),
            "store_open": pd.Series(True, index=range(len(dates)), dtype="boolean"),
        }
    )
    target_dates = dates[28:]
    weather = pd.DataFrame(
        {
            "issued_at": pd.Timestamp("2025-01-27", tz="Asia/Ho_Chi_Minh"),
            "target_date": target_dates,
            "temperature": 30.0,
            "rain": 0.0,
            "humidity": 70.0,
            "source": "previous_runs_fixture",
        }
    )
    bundle = replace(
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
    folds = [
        ForecastFold(
            name="validation_01",
            forecast_origin=dates[27],
            train_start=dates[0],
            train_end=dates[27],
            target_dates=tuple(dates[28:35]),
            role="validation",
        ),
        ForecastFold(
            name="validation_02",
            forecast_origin=dates[34],
            train_start=dates[0],
            train_end=dates[34],
            target_dates=tuple(dates[35:42]),
            role="validation",
        ),
    ]
    return bundle, folds
