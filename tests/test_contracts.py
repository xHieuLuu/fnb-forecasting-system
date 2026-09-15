from datetime import datetime

import pandas as pd
import pytest

from fnb_forecast.contracts import (
    DataBundle,
    DataProvenance,
    ForecastModel,
    ForecastRequest,
    assert_forecast_frame,
)
from fnb_forecast.exceptions import DataValidationError


def test_forecast_contract_rejects_missing_item_id() -> None:
    frame = pd.DataFrame(
        {
            "forecast_origin": ["2025-01-01"],
            "target_date": ["2025-01-02"],
            "yhat": [1.0],
            "model_id": ["naive"],
        }
    )
    try:
        assert_forecast_frame(frame)
    except DataValidationError as exc:
        assert "item_id" in str(exc)
    else:
        raise AssertionError("missing item_id must fail")


def test_forecast_request_rejects_non_seven_day_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_days must be 7"):
        ForecastRequest(
            forecast_origin=pd.Timestamp("2025-01-01"),
            horizon_days=14,
            dataset_name="daily_sales",
        )


def test_data_bundle_uses_the_approved_shared_dataset_fields() -> None:
    frame = pd.DataFrame()
    provenance = DataProvenance(
        name="synthetic-v1",
        source="generator",
        is_synthetic=True,
        generated_at=datetime(2026, 8, 13),
        checksum=None,
    )

    bundle = DataBundle(
        daily_sales=frame,
        item_master=frame,
        calendar=frame,
        weather=frame,
        recipes=frame,
        ingredient_master=frame,
        inventory_lots=frame,
        scheduled_receipts=frame,
        provenance=provenance,
    )

    assert bundle.provenance.name == "synthetic-v1"


def test_forecast_model_protocol_declares_adapter_methods() -> None:
    declared_methods = set(ForecastModel.__dict__)

    assert {"fit", "predict", "save"}.issubset(declared_methods)
