from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from fnb_forecast.config import ProjectConfig
from fnb_forecast.contracts import DataBundle, DataProvenance
from fnb_forecast.data.synthetic import SyntheticScenarioDefinition, load_synthetic_definition


@pytest.fixture
def project_config() -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "forecast": {"horizon_days": 7, "input_windows": [28, 56]},
            "scenario": {
                "start_date": "2024-01-02",
                "end_date": "2025-12-31",
                "seed": 20260813,
                "item_count": 15,
                "ingredient_count": 15,
            },
        }
    )


@pytest.fixture
def observed_weather(project_config: ProjectConfig) -> pd.DataFrame:
    """Deterministic historical weather for synthetic-target generation only."""
    dates = pd.date_range(
        project_config.scenario.start_date, project_config.scenario.end_date, freq="D"
    )
    offset = pd.Series(range(len(dates)), dtype="float64")
    return pd.DataFrame(
        {
            "date": dates,
            "temperature": 27.0 + (offset % 5) * 0.4,
            "rain": (offset % 4) * 1.5,
            "humidity": 68.0 + (offset % 7),
            "source": "historical_observation_fixture",
        }
    )


@pytest.fixture
def scenario_definition() -> SyntheticScenarioDefinition:
    return load_synthetic_definition(Path("configs/synthetic.yaml"))


@pytest.fixture
def valid_bundle() -> DataBundle:
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    return DataBundle(
        daily_sales=pd.DataFrame(
            {
                "date": dates,
                "item_id": ["coffee"] * 3,
                "quantity": [4, 3, 5],
                "unit_price": [30000, 30000, 30000],
                "promo_flag": pd.Series([False, False, False], dtype="boolean"),
                "stockout_flag": pd.Series([False, False, False], dtype="boolean"),
                "store_open": pd.Series([True, True, True], dtype="boolean"),
            }
        ),
        item_master=pd.DataFrame(
            {
                "item_id": ["coffee"],
                "item_name": ["Coffee"],
                "category": ["beverage"],
                "launch_date": [dates[0]],
                "active": pd.Series([True], dtype="boolean"),
            }
        ),
        calendar=pd.DataFrame(
            {
                "date": dates,
                "weekday": [2, 3, 4],
                "is_weekend": pd.Series([False, False, False], dtype="boolean"),
                "holiday_name": [None, None, None],
                "days_to_tet": [30, 29, 28],
                "days_after_tet": [None, None, None],
                "store_open": pd.Series([True, True, True], dtype="boolean"),
            }
        ),
        weather=pd.DataFrame(
            {
                "issued_at": [dates[0]] * 3,
                "target_date": dates,
                "temperature": [25.0, 26.0, 27.0],
                "rain": [0.0, 1.0, 0.0],
                "humidity": [70.0, 71.0, 72.0],
                "source": ["fixture"] * 3,
            }
        ),
        recipes=pd.DataFrame(
            {
                "item_id": ["coffee"],
                "ingredient_id": ["beans"],
                "amount": [0.02],
                "unit": ["kg"],
                "waste_rate": [0.0],
            }
        ),
        ingredient_master=pd.DataFrame(
            {
                "ingredient_id": ["beans"],
                "name": ["Coffee beans"],
                "base_unit": ["kg"],
                "pack_size": [1.0],
                "lead_time_days": [2],
                "review_period_days": [7],
                "shelf_life_days": [90],
            }
        ),
        inventory_lots=pd.DataFrame(
            {
                "ingredient_id": ["beans"],
                "lot_id": ["lot-1"],
                "on_hand": [2.0],
                "expiry_date": [pd.Timestamp("2025-03-01")],
                "received_date": [pd.Timestamp("2024-12-01")],
            }
        ),
        scheduled_receipts=pd.DataFrame(
            {
                "ingredient_id": pd.Series(dtype="object"),
                "arrival_date": pd.Series(dtype="datetime64[ns]"),
                "quantity": pd.Series(dtype="float64"),
            }
        ),
        provenance=DataProvenance(
            name="fixture",
            source="test",
            is_synthetic=False,
            generated_at=datetime(2025, 1, 1),
            checksum="fixture-checksum",
        ),
    )
