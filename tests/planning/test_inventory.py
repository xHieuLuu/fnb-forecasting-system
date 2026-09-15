import numpy as np
import pandas as pd
import pytest

from fnb_forecast.planning.inventory import (
    build_ingredient_residuals,
    compute_safety_stock,
    suggest_orders,
)


def ingredient_master(
    *,
    pack_size: float = 1.0,
    lead_time_days: int = 1,
    review_period_days: int = 1,
    shelf_life_days: int | None = 10,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ingredient_id": ["milk"],
            "pack_size": [pack_size],
            "lead_time_days": [lead_time_days],
            "review_period_days": [review_period_days],
            "shelf_life_days": [shelf_life_days],
            "perishable": [True],
        }
    )


def test_safety_stock_uses_underforecast_validation_errors_only() -> None:
    residuals = pd.DataFrame(
        {
            "role": ["validation"] * 4,
            "ingredient_id": ["milk"] * 4,
            "target_date": pd.date_range("2025-01-01", periods=4),
            "actual_need": [10, 14, 9, 20],
            "forecast_need": [12, 10, 9, 15],
        }
    )

    result = compute_safety_stock(residuals, ingredient_master(), service_level=0.95)

    assert np.isclose(result["safety_stock"].item(), 4.9)
    assert result["sample_count"].item() == 3


def test_order_excludes_expiring_stock_and_rounds_pack_size() -> None:
    need = pd.DataFrame(
        {
            "target_date": pd.date_range("2025-01-01", periods=2),
            "ingredient_id": ["milk", "milk"],
            "unit": ["L", "L"],
            "forecast_need": [10.0, 12.0],
        }
    )
    lots = pd.DataFrame(
        {
            "ingredient_id": ["milk", "milk"],
            "lot_id": ["usable", "expired"],
            "on_hand": [3.0, 10.0],
            "expiry_date": ["2025-01-20", "2024-12-31"],
            "received_date": ["2025-01-01", "2024-12-01"],
        }
    )
    receipts = pd.DataFrame(
        {"ingredient_id": ["milk"], "arrival_date": ["2025-01-01"], "quantity": [5.0]}
    )

    result = suggest_orders(
        need,
        pd.DataFrame({"ingredient_id": ["milk"], "safety_stock": [4.0]}),
        lots,
        receipts,
        ingredient_master(pack_size=6),
        order_date=pd.Timestamp("2025-01-01"),
    )

    assert result["usable_on_hand"].item() == 3
    assert result["raw_order"].item() == 18
    assert result["suggested_order"].item() == 18


def test_missing_perishable_shelf_life_disables_cap_with_warning() -> None:
    need = pd.DataFrame(
        {
            "target_date": ["2025-01-01"],
            "ingredient_id": ["milk"],
            "unit": ["L"],
            "forecast_need": [12.0],
        }
    )
    result = suggest_orders(
        need,
        pd.DataFrame({"ingredient_id": ["milk"], "safety_stock": [2.0]}),
        pd.DataFrame(
            {
                "ingredient_id": ["milk"],
                "lot_id": ["lot"],
                "on_hand": [0.0],
                "expiry_date": ["2025-01-10"],
                "received_date": ["2025-01-01"],
            }
        ),
        pd.DataFrame({"ingredient_id": [], "arrival_date": [], "quantity": []}),
        ingredient_master(shelf_life_days=None),
        order_date=pd.Timestamp("2025-01-01"),
    )

    assert not bool(result["shelf_life_cap_applied"].item())
    assert "Missing shelf life for perishable ingredient" in result["warning"].item()


def test_build_ingredient_residuals_rejects_test_rows() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": ["2025-01-02"],
            "item_id": ["milk_tea"],
            "actual": [10.0],
            "yhat": [8.0],
            "role": ["test"],
        }
    )
    recipes = pd.DataFrame(
        {
            "item_id": ["milk_tea"],
            "ingredient_id": ["milk"],
            "amount": [1.0],
            "unit": ["L"],
            "waste_rate": [0.0],
        }
    )

    with pytest.raises(ValueError, match="validation"):
        build_ingredient_residuals(predictions, recipes)
