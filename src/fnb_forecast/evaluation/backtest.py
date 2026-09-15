"""Leakage-safe rolling backtest execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from fnb_forecast.contracts import DataBundle, ForecastModel, assert_forecast_frame
from fnb_forecast.features import FeatureBuilder

from .splits import ForecastFold

FORECAST_KEY = ["forecast_origin", "target_date", "item_id"]
FUTURE_KNOWN_COLUMNS = ["target_date", "item_id", "unit_price", "promo_flag"]



class BacktestRunner:
    """Run fresh model and feature-builder instances for immutable folds."""


    def __init__(self, *, include_weather: bool = True) -> None:
        self.include_weather = include_weather

    def run(
        self,
        model_factory: Callable[[], ForecastModel],
        bundle: DataBundle,
        folds: Sequence[ForecastFold],
    ) -> pd.DataFrame:
        sales = bundle.daily_sales.copy()
        sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
        item_ids = bundle.item_master.loc[:, ["item_id"]].copy()
        if item_ids["item_id"].isna().any() or item_ids.duplicated("item_id").any():
            raise ValueError("item_master must contain unique, non-null item ids")
        fold_results: list[pd.DataFrame] = []

        total_folds = len(folds)
        for idx, fold in enumerate(folds, start=1):
            print(f"  [*] [Fold {idx}/{total_folds}] Evaluating fold '{fold.name}' ({fold.role})...")
            training_history = sales.loc[
                sales["date"].between(fold.train_start, fold.train_end)
            ].copy()
            as_of_history = sales.loc[
                sales["date"].between(fold.train_start, fold.forecast_origin)
            ].copy()
            target_dates = pd.DatetimeIndex(fold.target_dates).normalize()
            expected_item_dates = item_ids.merge(
                pd.DataFrame({"target_date": target_dates}), how="cross"
            )
            target_sales = sales.loc[sales["date"].isin(target_dates)].rename(
                columns={"date": "target_date"}
            )
            source_keys = target_sales.loc[:, ["target_date", "item_id"]]
            key_comparison = expected_item_dates.merge(
                source_keys,
                on=["target_date", "item_id"],
                how="outer",
                indicator=True,
            )
            if (
                source_keys.duplicated(["target_date", "item_id"]).any()
                or not key_comparison["_merge"].eq("both").all()
            ):
                raise ValueError(
                    "target-period daily_sales must contain exactly one row for every item "
                    "and target date"
                )
            builder = FeatureBuilder(
                calendar=bundle.calendar,
                weather=bundle.weather,
                include_weather=self.include_weather,
            )
            future_known = target_sales.loc[:, FUTURE_KNOWN_COLUMNS].copy()
            as_of_features, future_covariates = builder.build(
                as_of_history,
                future_known,
                fold.forecast_origin,
                fit_history=training_history,
            )
            as_of_features = self._add_model_columns(as_of_features, bundle)
            future_covariates = self._add_model_columns(future_covariates, bundle)
            training_features = as_of_features.loc[
                as_of_features["date"].le(fold.train_end)
            ].copy()

            model = model_factory()
            model.fit(training_features, future_covariates.iloc[0:0].copy())
            forecast = model.predict(as_of_features, future_covariates)
            assert_forecast_frame(forecast)
            forecast = forecast.copy()
            forecast["forecast_origin"] = pd.to_datetime(
                forecast["forecast_origin"], errors="raise"
            ).dt.normalize()
            forecast["target_date"] = pd.to_datetime(
                forecast["target_date"], errors="raise"
            ).dt.normalize()
            if forecast.duplicated(FORECAST_KEY).any():
                raise ValueError("forecast rows must be unique by forecast origin, date, and item")

            expected_keys = expected_item_dates.assign(
                forecast_origin=pd.Timestamp(fold.forecast_origin).normalize()
            )
            actual_keys = forecast.loc[:, FORECAST_KEY]
            if len(actual_keys) != len(expected_keys) or not (
                actual_keys.merge(
                    expected_keys,
                    on=FORECAST_KEY,
                    how="outer",
                    indicator=True,
                )["_merge"]
                .eq("both")
                .all()
            ):
                raise ValueError("forecast rows must match every requested item and target date")

            actuals = target_sales.loc[
                :, ["target_date", "item_id", "quantity"]
            ].rename(
                columns={"quantity": "actual"}
            )
            result = forecast.merge(
                actuals,
                on=["target_date", "item_id"],
                how="left",
                validate="one_to_one",
            )
            result["fold_name"] = fold.name
            result["role"] = fold.role
            result["dataset_name"] = bundle.provenance.name
            fold_results.append(result)

        if not fold_results:
            return pd.DataFrame(
                columns=[
                    *FORECAST_KEY,
                    "yhat",
                    "model_id",
                    "actual",
                    "fold_name",
                    "role",
                    "dataset_name",
                ]
            )
        combined = pd.concat(fold_results, ignore_index=True)
        if combined.duplicated(FORECAST_KEY).any():
            raise ValueError("backtest rows must be unique by forecast origin, date, and item")
        return combined

    @staticmethod
    def _add_model_columns(frame: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
        """Add static/calendar columns required by global adapters."""
        result = frame.copy()
        categories = bundle.item_master.set_index("item_id")["category"]
        if "category" not in result.columns:
            result["category"] = result["item_id"].map(categories).fillna("unknown")
        if "month" not in result.columns:
            date_column = "target_date" if "target_date" in result.columns else "date"
            result["month"] = pd.to_datetime(result[date_column]).dt.month.astype("float64")
        for column in ("temperature", "rain", "humidity"):
            if column not in result.columns:
                result[column] = 0.0
        return result
