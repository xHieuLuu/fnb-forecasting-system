"""Inference orchestration with explicit, auditable fallback ordering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from fnb_forecast.contracts import DataBundle, ForecastRequest, assert_forecast_frame
from fnb_forecast.data.validation import validate_bundle
from fnb_forecast.exceptions import (
    ArtifactCompatibilityError,
    ModelInferenceError,
    WeatherUnavailableError,
)
from fnb_forecast.models.naive import SeasonalNaiveModel
from fnb_forecast.planning.inventory import (
    build_ingredient_residuals,
    compute_safety_stock,
    suggest_orders,
)
from fnb_forecast.planning.revenue import (
    PlanningOutput,
    calculate_ingredient_need,
    calculate_revenue,
)


class ForecastService:
    """Run selected inference and planning while keeping fallback causes visible."""

    def __init__(
        self,
        *,
        registry: Any,
        feature_builder: Any,
        selected_model_id: str,
        selected_run_id: str,
        no_weather_feature_builder: Any | None = None,
        no_weather_model_id: str | None = None,
        no_weather_run_id: str | None = None,
        model_factory: Callable[[str], Any] | None = None,
        interval_calibrator: Any | None = None,
    ) -> None:
        self.registry = registry
        self.feature_builder = feature_builder
        self.selected_model_id = selected_model_id
        self.selected_run_id = selected_run_id
        self.no_weather_feature_builder = no_weather_feature_builder
        self.no_weather_model_id = no_weather_model_id
        self.no_weather_run_id = no_weather_run_id
        self.model_factory = model_factory
        self.interval_calibrator = interval_calibrator

    def run(
        self, request: ForecastRequest, bundle: DataBundle, service_level: float
    ) -> PlanningOutput:
        if service_level not in {0.90, 0.95, 0.98}:
            raise ValueError("service_level must be one of 0.90, 0.95, or 0.98")

        checked = validate_bundle(bundle)
        warnings: list[str] = []
        requested_model = self.selected_model_id
        actual_model = requested_model
        fallback_reason: str | None = None
        history_source = self._history_as_of(checked.daily_sales, request)
        future_known = self._future_known(checked, request)
        use_seasonal_naive = False

        try:
            history, future = self.feature_builder.build(
                history_source,
                future_known,
                request.forecast_origin,
                fit_history=history_source,
            )
        except WeatherUnavailableError as error:
            fallback_reason = f"weather unavailable: {error}"
            if self.no_weather_feature_builder is not None and self.no_weather_model_id is not None:
                try:
                    warnings.append(f"Weather unavailable; used no-weather variant: {error}")
                    actual_model = self.no_weather_model_id
                    history, future = self.no_weather_feature_builder.build(
                        history_source,
                        future_known,
                        request.forecast_origin,
                        fit_history=history_source,
                    )
                except WeatherUnavailableError as no_weather_error:
                    fallback_reason = (
                        f"{fallback_reason}; no-weather variant unavailable: {no_weather_error}"
                    )
                    warnings.append(
                        f"Weather unavailable; used seasonal_naive_7: {no_weather_error}"
                    )
                    history, future = history_source, future_known
                    actual_model = "seasonal_naive_7"
                    use_seasonal_naive = True
            else:
                warnings.append(f"Weather unavailable; used seasonal_naive_7: {error}")
                history, future = history_source, future_known
                actual_model = "seasonal_naive_7"
                use_seasonal_naive = True

        if use_seasonal_naive:
            model = self._fallback_model(history_source)
            predictions = self._predict(model, history_source, future_known)
        else:
            from fnb_forecast.evaluation.backtest import BacktestRunner

            history = BacktestRunner._add_model_columns(history, checked)
            future = BacktestRunner._add_model_columns(future, checked)
            try:
                model = self._load_model(
                    request.dataset_name, actual_model, self._run_id(actual_model)
                )
                predictions = self._predict(model, history, future)
            except (ArtifactCompatibilityError, ModelInferenceError) as error:
                failed_model = actual_model
                model = self._fallback_model(history_source)
                actual_model = model.model_id
                reason = f"{failed_model}: {error}"
                fallback_reason = f"{fallback_reason}; {reason}" if fallback_reason else reason
                warnings.append("Selected artifact unavailable; used seasonal_naive_7")
                predictions = self._predict(model, history_source, future_known)

        predictions = self._add_audit_columns(
            predictions,
            request=request,
            requested_model=requested_model,
            actual_model=actual_model,
            fallback_reason=fallback_reason,
            service_level=service_level,
        )
        predictions, new_item_warning = self._apply_new_item_fallback(
            predictions, history_source, checked
        )
        if new_item_warning:
            warnings.append(new_item_warning)
        if self.interval_calibrator is not None:
            predictions = self._add_interval_context(predictions, checked)
            predictions = self.interval_calibrator.apply(predictions)

        prices = self._future_prices(checked, request)
        revenue = calculate_revenue(predictions, prices)
        ingredient_need = calculate_ingredient_need(predictions, checked.recipes)
        safety_stock = self._safety_stock(
            checked,
            request,
            warnings,
            model_id=actual_model,
            service_level=service_level,
        )
        orders = suggest_orders(
            ingredient_need.totals,
            safety_stock,
            checked.inventory_lots,
            checked.scheduled_receipts,
            checked.ingredient_master,
            order_date=request.forecast_origin,
        )
        fallback_reasons = [reason for reason in (fallback_reason, new_item_warning) if reason]
        metadata = {
            "requested_model": requested_model,
            "actual_model": actual_model,
            "fallback_used": bool(fallback_reasons),
            "fallback_reason": "; ".join(fallback_reasons) or None,
            "forecast_origin": str(pd.Timestamp(request.forecast_origin).normalize()),
            "dataset_name": request.dataset_name,
            "dataset_checksum": checked.provenance.checksum,
            "provenance": checked.provenance.name,
            "horizon_days": request.horizon_days,
            "service_level": service_level,
            "warnings": warnings + ingredient_need.warnings,
        }
        return PlanningOutput(
            item_forecasts=predictions,
            revenue_forecast=revenue,
            ingredient_forecast=ingredient_need.totals,
            ingredient_detail=ingredient_need.detail,
            order_proposals=orders,
            warnings=warnings + ingredient_need.warnings,
            metadata=metadata,
        )

    def _load_model(self, dataset_name: str, model_id: str, run_id: str) -> Any:
        if self.model_factory is not None:
            return self.model_factory(model_id)
        return self.registry.load(dataset_name, model_id, run_id)

    def _run_id(self, model_id: str) -> str:
        if model_id == self.no_weather_model_id and self.no_weather_run_id:
            return self.no_weather_run_id
        return self.selected_run_id

    @staticmethod
    def _predict(model: Any, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        try:
            predictions = model.predict(history, future)
            assert_forecast_frame(predictions)
            if not np.isfinite(predictions["yhat"].to_numpy(dtype="float64")).all():
                raise ModelInferenceError("model produced non-finite forecasts")
            return predictions
        except ModelInferenceError:
            raise
        except Exception as error:
            raise ModelInferenceError(str(error)) from error

    @staticmethod
    def _fallback_model(history: pd.DataFrame) -> SeasonalNaiveModel:
        model = SeasonalNaiveModel()
        model.fit(history, history.iloc[0:0])
        return model

    @staticmethod
    def _add_audit_columns(
        predictions: pd.DataFrame,
        *,
        request: ForecastRequest,
        requested_model: str,
        actual_model: str,
        fallback_reason: str | None,
        service_level: float,
    ) -> pd.DataFrame:
        result = predictions.copy()
        result["forecast_origin"] = pd.to_datetime(result["forecast_origin"]).dt.normalize()
        result["target_date"] = pd.to_datetime(result["target_date"]).dt.normalize()
        result["requested_model"] = requested_model
        result["actual_model"] = actual_model
        result["fallback_used"] = bool(fallback_reason)
        result["fallback_reason"] = fallback_reason or ""
        result["service_level"] = service_level
        expected_origin = ForecastService._local_day(request.forecast_origin)
        if not result["forecast_origin"].eq(expected_origin).all():
            raise ModelInferenceError("model returned a forecast origin different from request")
        return result

    @staticmethod
    def _add_interval_context(predictions: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
        result = predictions.copy()
        result["horizon"] = (
            pd.to_datetime(result["target_date"]) - pd.to_datetime(result["forecast_origin"])
        ).dt.days.astype("int64")
        if "category" not in result.columns:
            categories = bundle.item_master.set_index("item_id")["category"]
            result["category"] = result["item_id"].map(categories).fillna("__all__")
        return result

    @staticmethod
    def _future_known(bundle: DataBundle, request: ForecastRequest) -> pd.DataFrame:
        origin = ForecastService._local_day(request.forecast_origin)
        dates = pd.date_range(origin + pd.Timedelta(1, unit="D"), periods=request.horizon_days)
        items = bundle.item_master.loc[:, ["item_id"]].drop_duplicates()
        future = items.merge(pd.DataFrame({"target_date": dates}), how="cross")
        prices = bundle.daily_sales.loc[:, ["date", "item_id", "unit_price", "promo_flag"]].rename(
            columns={"date": "target_date"}
        )
        prices["target_date"] = pd.to_datetime(prices["target_date"], errors="raise").dt.normalize()
        merged = future.merge(prices, on=["target_date", "item_id"], how="left")
        last_prices = bundle.daily_sales.groupby("item_id")["unit_price"].last()
        merged["unit_price"] = merged["unit_price"].fillna(merged["item_id"].map(last_prices))
        merged["promo_flag"] = merged["promo_flag"].fillna(False)
        return merged

    @staticmethod
    def _future_prices(bundle: DataBundle, request: ForecastRequest) -> pd.DataFrame:
        return ForecastService._future_known(bundle, request).loc[
            :, ["target_date", "item_id", "unit_price"]
        ]

    def _safety_stock(
        self,
        bundle: DataBundle,
        request: ForecastRequest,
        warnings: list[str],
        *,
        model_id: str,
        service_level: float,
    ) -> pd.DataFrame:
        try:
            validation = self.registry.load_validation_predictions(request.dataset_name, model_id)
            residuals = build_ingredient_residuals(validation, bundle.recipes)
            return compute_safety_stock(residuals, bundle.ingredient_master, service_level)
        except (ArtifactCompatibilityError, ValueError, KeyError) as error:
            warnings.append(f"Safety stock unavailable; used zero safety stock: {error}")
            return bundle.ingredient_master.loc[:, ["ingredient_id"]].assign(safety_stock=0.0)

    @staticmethod
    def _apply_new_item_fallback(
        predictions: pd.DataFrame,
        history: pd.DataFrame,
        bundle: DataBundle,
    ) -> tuple[pd.DataFrame, str | None]:
        sales = history.copy()
        sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
        observed = sales.loc[sales["quantity"].notna()]
        counts = observed.groupby("item_id")["date"].nunique()
        new_items = set(counts[counts < 7].index)
        new_items.update(set(bundle.item_master["item_id"]).difference(counts.index))
        if not new_items:
            return predictions, None
        category_by_item = bundle.item_master.set_index("item_id")["category"].to_dict()
        observed_with_category = observed.assign(
            category=observed["item_id"].map(category_by_item)
        )
        medians = observed_with_category.groupby("category")["quantity"].median()
        result = predictions.copy()
        for item_id in new_items:
            category = category_by_item.get(item_id)
            median_value = medians.get(category, observed["quantity"].median())
            median = float(median_value if pd.notna(median_value) else 0.0)
            mask = result["item_id"].eq(item_id)
            result.loc[mask, "yhat"] = median
            result.loc[mask, "model_id"] = "category_median"
            result.loc[mask, "actual_model"] = "category_median"
            result.loc[mask, "fallback_used"] = True
            prior_reason = result.loc[mask, "fallback_reason"].astype(str)
            new_reason = "new item has fewer than seven history days"
            result.loc[mask, "fallback_reason"] = prior_reason.map(
                lambda reason: f"{reason}; {new_reason}" if reason else new_reason
            )
        return result, f"Used category median for new items: {sorted(new_items, key=str)}"

    @staticmethod
    def _history_as_of(sales: pd.DataFrame, request: ForecastRequest) -> pd.DataFrame:
        history = sales.copy()
        history["date"] = pd.to_datetime(history["date"], errors="raise").dt.normalize()
        origin = ForecastService._local_day(request.forecast_origin)
        history = history.loc[history["date"].le(origin)].copy()
        if history.empty:
            raise ValueError("daily_sales has no observations at or before forecast origin")
        return history.sort_values(["item_id", "date"]).reset_index(drop=True)

    @staticmethod
    def _local_day(value: pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
        return timestamp.normalize()
