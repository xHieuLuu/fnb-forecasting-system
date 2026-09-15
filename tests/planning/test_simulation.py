import pandas as pd

from fnb_forecast.planning.simulation import InventorySimulator


def test_simulation_reports_both_policies_and_never_uses_expired_stock() -> None:
    actual = pd.DataFrame(
        {
            "target_date": pd.date_range("2025-01-01", periods=3),
            "ingredient_id": ["milk"] * 3,
            "actual_need": [2.0, 2.0, 2.0],
        }
    )
    forecast = actual.rename(columns={"actual_need": "forecast_need"}).copy()
    lots = pd.DataFrame(
        {
            "ingredient_id": ["milk"],
            "lot_id": ["expired-before-start"],
            "on_hand": [10.0],
            "expiry_date": ["2024-12-31"],
            "received_date": ["2024-12-01"],
        }
    )
    master = pd.DataFrame(
        {
            "ingredient_id": ["milk"],
            "pack_size": [1.0],
            "lead_time_days": [5],
            "review_period_days": [1],
            "shelf_life_days": [10],
            "perishable": [True],
        }
    )

    result = InventorySimulator(
        actual_need=actual,
        model_forecasts=forecast,
        inventory_lots=lots,
        scheduled_receipts=pd.DataFrame(columns=["ingredient_id", "arrival_date", "quantity"]),
        ingredient_master=master,
    ).run()

    assert set(result.policy_summary["policy"]) == {"forecast", "baseline"}
    assert set(result.daily_audit["policy"]) == {"forecast", "baseline"}
    assert result.policy_summary["stockout_days"].ge(1).all()
    assert result.daily_audit["served"].sum() == 0
