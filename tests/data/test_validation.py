import pandas as pd
import pytest

from fnb_forecast.data.validation import complete_daily_sales, validate_bundle, validate_daily_sales
from fnb_forecast.exceptions import DataValidationError


def test_negative_quantity_reports_rows_and_column() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "item_id": ["coffee"],
            "quantity": [-1],
            "unit_price": [30000],
            "promo_flag": [False],
            "stockout_flag": [False],
            "store_open": [True],
        }
    )

    with pytest.raises(DataValidationError) as caught:
        validate_daily_sales(sales)

    assert caught.value.columns == ["quantity"]
    assert caught.value.rows == [0]


def test_closed_store_quantity_is_rejected() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "item_id": ["coffee"],
            "quantity": [4],
            "unit_price": [30000],
            "promo_flag": [False],
            "stockout_flag": [False],
            "store_open": [False],
        }
    )

    with pytest.raises(DataValidationError) as caught:
        validate_daily_sales(sales)

    assert caught.value.columns == ["quantity"]
    assert caught.value.rows == [0]


def test_unknown_store_open_is_rejected() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "item_id": ["coffee"],
            "quantity": [4],
            "unit_price": [30000],
            "promo_flag": [False],
            "stockout_flag": [False],
            "store_open": [None],
        }
    )

    with pytest.raises(DataValidationError) as caught:
        validate_daily_sales(sales)

    assert caught.value.columns == ["store_open"]
    assert caught.value.rows == [0]


def test_open_missing_day_becomes_zero_but_closed_day_stays_closed() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "item_id": ["coffee"],
            "quantity": [4],
            "unit_price": [30000],
            "promo_flag": [False],
            "stockout_flag": [False],
            "store_open": [True],
        }
    )
    items = pd.DataFrame({"item_id": ["coffee"]})
    calendar = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "store_open": [True, True, False],
        }
    )

    result = complete_daily_sales(sales, items, calendar)

    assert result.loc[result["date"].eq("2025-01-02"), "quantity"].item() == 0
    assert pd.isna(result.loc[result["date"].eq("2025-01-03"), "quantity"].item())


def test_duplicate_normalized_calendar_days_are_rejected() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "item_id": ["coffee"],
            "quantity": [4],
            "unit_price": [30000],
            "promo_flag": [False],
            "stockout_flag": [False],
            "store_open": [True],
        }
    )
    items = pd.DataFrame({"item_id": ["coffee"]})
    calendar = pd.DataFrame(
        {
            "date": ["2025-01-02 09:00:00", "2025-01-02 17:00:00"],
            "store_open": [True, False],
        }
    )

    with pytest.raises(DataValidationError) as caught:
        complete_daily_sales(sales, items, calendar)

    assert caught.value.table == "calendar"
    assert caught.value.columns == ["date"]
    assert caught.value.rows == [0, 1]


def test_validate_bundle_normalizes_dates_and_nullable_booleans(valid_bundle) -> None:
    valid_bundle.daily_sales.loc[0, "date"] = "2025-01-01 18:00:00"
    valid_bundle.daily_sales.loc[0, "promo_flag"] = None

    result = validate_bundle(valid_bundle)

    assert result.daily_sales["date"].iloc[0] == pd.Timestamp("2025-01-01")
    assert str(result.daily_sales["promo_flag"].dtype) == "boolean"
