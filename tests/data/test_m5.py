from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fnb_forecast.cli import app
from fnb_forecast.data.m5 import select_food_items

M5_FIXTURE = Path("tests/fixtures/m5")


def test_item_selection_uses_only_dates_before_selection_end() -> None:
    sales = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
            "item_id": ["stable", "stable", "future_spike", "future_spike"],
            "category": ["FOODS"] * 4,
            "quantity": [5, 5, 1000, 1000],
        }
    )

    selected = select_food_items(sales, pd.Timestamp("2024-01-15"), n_items=1)

    assert selected == ["stable"]


def test_load_m5_normalizes_selected_store_with_prices_and_empty_supply_tables() -> None:
    from fnb_forecast.data.m5 import load_m5

    bundle = load_m5(M5_FIXTURE, "CA_1", n_items=2, selection_end=pd.Timestamp("2024-01-02"))

    assert list(bundle.daily_sales[["item_id", "date"]].itertuples(index=False, name=None)) == [
        ("FOODS_1_001", pd.Timestamp("2024-01-01")),
        ("FOODS_1_001", pd.Timestamp("2024-01-02")),
        ("FOODS_1_001", pd.Timestamp("2024-01-03")),
        ("FOODS_1_002", pd.Timestamp("2024-01-01")),
        ("FOODS_1_002", pd.Timestamp("2024-01-02")),
        ("FOODS_1_002", pd.Timestamp("2024-01-03")),
    ]
    assert bundle.daily_sales.loc[
        bundle.daily_sales["item_id"].eq("FOODS_1_001"), "unit_price"
    ].tolist() == [1.25, 1.25, 1.5]
    assert bundle.recipes.empty
    assert bundle.ingredient_master.empty
    assert bundle.inventory_lots.empty
    assert bundle.scheduled_receipts.empty
    assert bundle.provenance.name == "m5_public_benchmark"
    assert bundle.provenance.is_synthetic is False


def test_prepare_m5_command_saves_the_offline_fixture_bundle(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "prepare-m5",
            "--raw-dir",
            str(M5_FIXTURE),
            "--selection-end",
            "2024-01-02",
            "--n-items",
            "2",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "daily_sales.csv").exists()
    assert (tmp_path / "provenance.json").exists()
