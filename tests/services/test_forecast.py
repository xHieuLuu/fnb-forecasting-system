from dataclasses import replace

import pandas as pd
import pytest

from fnb_forecast.contracts import ForecastRequest
from fnb_forecast.exceptions import ArtifactCompatibilityError, WeatherUnavailableError
from fnb_forecast.services.forecast import ForecastService


class FakeBuilder:
    def __init__(self, *, weather_failure: bool = False) -> None:
        self.weather_failure = weather_failure

    def build(self, history: pd.DataFrame, future: pd.DataFrame, origin: pd.Timestamp, **kwargs):
        del origin, kwargs
        if self.weather_failure:
            raise WeatherUnavailableError("weather unavailable")
        return history.copy(), future.copy()


class FakeModel:
    def __init__(self, model_id: str, *, nan_output: bool = False) -> None:
        self.model_id = model_id
        self.nan_output = nan_output

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        del history, known_covariates

    def predict(self, history: pd.DataFrame, future_covariates: pd.DataFrame) -> pd.DataFrame:
        del history
        result = future_covariates.loc[:, ["target_date", "item_id"]].copy()
        result["forecast_origin"] = result["target_date"].min() - pd.Timedelta(1, unit="D")
        result["yhat"] = float("nan") if self.nan_output else 5.0
        result["model_id"] = self.model_id
        return result.loc[:, ["forecast_origin", "target_date", "item_id", "yhat", "model_id"]]


class FakeRegistry:
    def __init__(self, models: dict[str, object] | None = None) -> None:
        self.models = models or {}

    def load(self, dataset_name: str, model_id: str, run_id: str, **kwargs):
        del dataset_name, run_id, kwargs
        if model_id not in self.models:
            raise ArtifactCompatibilityError("artifact unavailable")
        return self.models[model_id]

    def load_validation_predictions(self, dataset_name: str, model_id: str) -> pd.DataFrame:
        del dataset_name, model_id
        raise ArtifactCompatibilityError("validation artifact unavailable")


def request() -> ForecastRequest:
    return ForecastRequest(
        forecast_origin=pd.Timestamp("2025-01-07"), horizon_days=7, dataset_name="synthetic"
    )


def test_missing_selected_artifact_falls_back_to_seasonal_naive(service_bundle) -> None:
    service = ForecastService(
        registry=FakeRegistry(),
        feature_builder=FakeBuilder(),
        selected_model_id="lstm_full",
        selected_run_id="run-1",
    )

    result = service.run(request(), service_bundle, service_level=0.95)

    assert result.item_forecasts["model_id"].unique().tolist() == ["seasonal_naive_7"]
    assert result.item_forecasts["fallback_used"].all()
    assert "Selected artifact unavailable; used seasonal_naive_7" in result.warnings


def test_weather_failure_uses_registered_no_weather_variant(service_bundle) -> None:
    service = ForecastService(
        registry=FakeRegistry({"lstm_history_calendar": FakeModel("lstm_history_calendar")}),
        feature_builder=FakeBuilder(weather_failure=True),
        no_weather_feature_builder=FakeBuilder(),
        selected_model_id="lstm_full",
        selected_run_id="run-full",
        no_weather_model_id="lstm_history_calendar",
        no_weather_run_id="run-calendar",
    )

    result = service.run(request(), service_bundle, service_level=0.95)

    assert result.item_forecasts["model_id"].unique().tolist() == ["lstm_history_calendar"]
    assert any("weather" in warning.lower() for warning in result.warnings)


def test_invalid_model_output_falls_back_and_records_failed_model(service_bundle) -> None:
    service = ForecastService(
        registry=FakeRegistry({"bad_model": FakeModel("bad_model", nan_output=True)}),
        feature_builder=FakeBuilder(),
        selected_model_id="bad_model",
        selected_run_id="run-bad",
    )

    result = service.run(request(), service_bundle, service_level=0.95)

    assert result.item_forecasts["model_id"].unique().tolist() == ["seasonal_naive_7"]
    assert "bad_model" in result.metadata["fallback_reason"]


def test_new_item_uses_category_median_only_for_short_history(service_bundle) -> None:
    bundle = service_bundle
    new_dates = pd.date_range("2025-01-05", periods=3)
    new_sales = bundle.daily_sales.iloc[:3].copy()
    new_sales["date"] = new_dates
    new_sales["item_id"] = "tea"
    new_sales["quantity"] = [8, 10, 12]
    sales = pd.concat([bundle.daily_sales, new_sales], ignore_index=True)
    items = pd.concat(
        [
            bundle.item_master,
            pd.DataFrame(
                {
                    "item_id": ["tea"],
                    "item_name": ["Tea"],
                    "category": ["beverage"],
                    "launch_date": [new_dates[0]],
                    "active": pd.Series([True], dtype="boolean"),
                }
            ),
        ],
        ignore_index=True,
    )
    bundle = replace(bundle, daily_sales=sales, item_master=items)
    service = ForecastService(
        registry=FakeRegistry({"lstm_full": FakeModel("lstm_full")}),
        feature_builder=FakeBuilder(),
        selected_model_id="lstm_full",
        selected_run_id="run-full",
    )

    result = service.run(request(), bundle, service_level=0.95)
    tea = result.item_forecasts.loc[result.item_forecasts["item_id"].eq("tea")]
    coffee = result.item_forecasts.loc[result.item_forecasts["item_id"].eq("coffee")]

    assert tea["model_id"].unique().tolist() == ["category_median"]
    assert tea["yhat"].eq(4.5).all()
    assert coffee["model_id"].unique().tolist() == ["lstm_full"]
    assert result.metadata["fallback_used"] is True


@pytest.fixture
def service_bundle(valid_bundle):
    dates = pd.date_range("2025-01-01", periods=7)
    sales = valid_bundle.daily_sales.iloc[[0] * 7].copy()
    sales["date"] = dates
    sales["quantity"] = [4, 3, 5, 4, 3, 5, 4]
    return replace(valid_bundle, daily_sales=sales)
