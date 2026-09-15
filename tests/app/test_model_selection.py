"""Tests for model selection functionality in dashboard state."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from fnb_forecast.app.state import get_model_options, run_forecast
from fnb_forecast.cli import _load_interval_calibrator
from fnb_forecast.planning.revenue import PlanningOutput


def test_get_model_options_contains_auto_and_registered_models() -> None:
    options = get_model_options("real_store_14m")
    assert "auto" in options
    assert "Tự động" in options["auto"]
    assert "lgb_l15_d-1_lr0.03" in options
    assert "seasonal_naive_7" in options


def test_run_forecast_accepts_model_id_parameter() -> None:
    # Test that run_forecast accepts model_id and passes it to ForecastService
    with patch("fnb_forecast.app.state.ForecastService") as mock_service_class:
        mock_service = MagicMock()
        mock_service.run.return_value = PlanningOutput(
            item_forecasts=pd.DataFrame(),
            revenue_forecast=pd.DataFrame(),
            ingredient_forecast=pd.DataFrame(),
            ingredient_detail=pd.DataFrame(),
            order_proposals=pd.DataFrame(),
            warnings=[],
            metadata={"requested_model": "seasonal_naive_7", "actual_model": "seasonal_naive_7"},
        )
        mock_service_class.return_value = mock_service

        output = run_forecast(
            origin=pd.Timestamp("2025-12-24"),
            service_level=0.95,
            dataset="real_store_14m",
            model_id="seasonal_naive_7",
        )

        assert mock_service_class.called
        kwargs = mock_service_class.call_args.kwargs
        assert kwargs["selected_model_id"] == "seasonal_naive_7"
        assert output.metadata["requested_model"] == "seasonal_naive_7"


def test_registered_models_execute_without_fallback() -> None:
    # Test real execution of registered models on real_store_14m
    for model_id in ["lgb_l15_d-1_lr0.03", "lgb_l15_d-1_lr0.05", "auto_ets_7", "seasonal_naive_7"]:
        out = run_forecast(
            origin=pd.Timestamp("2026-06-23"),
            service_level=0.95,
            dataset="real_store_14m",
            model_id=model_id,
        )
        assert out.metadata["requested_model"] == model_id
        assert out.metadata["actual_model"] == model_id
        assert out.metadata["fallback_used"] is False


def test_explicit_model_uses_that_models_interval_calibrator(tmp_path) -> None:
    """An explicit model choice must not reuse the frozen auto-selection calibrator."""
    registry_root = tmp_path / "registry"
    model_root = registry_root / "real_store_calibrated_14m" / "lgb_model" / "lgb_model-validation"
    model_root.mkdir(parents=True)
    (model_root / "interval_state.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        {
            "actual": [2.0, 4.0],
            "yhat": [1.0, 3.0],
            "role": ["validation", "validation"],
            "horizon": [1, 1],
        }
    ).to_parquet(model_root / "validation_predictions.parquet")

    calibrator = _load_interval_calibrator(
        registry_root,
        "real_store_calibrated_14m",
        {
            "selected_model_id": "lgb_model",
            "selected_run_id": "lgb_model-validation",
        },
    )

    assert calibrator is not None
    assert calibrator.state["quantiles"]
