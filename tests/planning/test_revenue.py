import pandas as pd
import pytest

from fnb_forecast.exceptions import DataValidationError
from fnb_forecast.planning.revenue import (
    calculate_ingredient_need,
    calculate_revenue,
)


def forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin": ["2025-01-01", "2025-01-01"],
            "target_date": ["2025-01-02", "2025-01-02"],
            "item_id": ["milk_tea", "unknown_item"],
            "yhat": [10.0, 2.0],
            "model_id": "lstm",
        }
    )


def one_item_forecast() -> pd.DataFrame:
    return forecast_frame().iloc[[0]].copy()


def one_item_recipe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["milk_tea", "milk_tea"],
            "ingredient_id": ["milk", "tea"],
            "amount": [120.0, 8.0],
            "unit": ["ml", "g"],
            "waste_rate": [0.05, 0.10],
        }
    )


def test_revenue_and_bom_conversion_are_consistent() -> None:
    prices = pd.DataFrame(
        {
            "target_date": ["2025-01-02"],
            "item_id": ["milk_tea"],
            "unit_price": [30_000],
        }
    )

    revenue = calculate_revenue(one_item_forecast(), prices)
    ingredients = calculate_ingredient_need(one_item_forecast(), one_item_recipe())

    assert revenue.loc[0, "forecast_revenue"] == 300_000
    totals = ingredients.totals.set_index("ingredient_id")
    assert totals.loc["milk", "forecast_need"] == 1260
    assert totals.loc["tea", "forecast_need"] == 88
    assert ingredients.warnings == []


def test_missing_bom_warns_and_excludes_only_that_item() -> None:
    ingredients = calculate_ingredient_need(forecast_frame(), one_item_recipe())

    assert ingredients.warnings == ["Missing BOM for item: unknown_item"]
    assert "unknown_item" not in ingredients.detail["source_item_id"].unique()


def test_duplicate_prices_raise_data_validation_error() -> None:
    prices = pd.DataFrame(
        {
            "target_date": ["2025-01-02", "2025-01-02"],
            "item_id": ["milk_tea", "milk_tea"],
            "unit_price": [30_000, 31_000],
        }
    )

    with pytest.raises(DataValidationError, match="prices"):
        calculate_revenue(one_item_forecast(), prices)


def test_bom_schema_rejects_bad_waste_and_mixed_units() -> None:
    bad_waste = one_item_recipe().assign(waste_rate=[-0.1, 0.1])
    with pytest.raises(DataValidationError, match="recipes"):
        calculate_ingredient_need(one_item_forecast(), bad_waste)

    mixed_units = one_item_recipe().assign(unit=["ml", "ml"])
    mixed_units.loc[1, "ingredient_id"] = "milk"
    with pytest.raises(DataValidationError, match="recipes"):
        calculate_ingredient_need(one_item_forecast(), mixed_units)


def test_missing_price_is_rejected_because_revenue_would_be_incomplete() -> None:
    prices = pd.DataFrame(
        {"target_date": ["2025-01-02"], "item_id": ["other"], "unit_price": [30_000]}
    )

    with pytest.raises(DataValidationError, match="prices"):
        calculate_revenue(one_item_forecast(), prices)
