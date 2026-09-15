"""As-of feature construction for seven-day forecasts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from fnb_forecast.exceptions import WeatherUnavailableError

LAGS = (1, 7, 14, 21, 28)
ROLLING_WINDOWS = (7, 14, 28)
CALENDAR_COLUMNS = (
    "weekday",
    "is_weekend",
    "holiday_name",
    "days_to_tet",
    "days_after_tet",
    "store_open",
)
WEATHER_NUMERIC_COLUMNS = ("temperature", "rain", "humidity")
WEATHER_REQUIRED_COLUMNS = {
    "issued_at",
    "target_date",
    *WEATHER_NUMERIC_COLUMNS,
    "source",
}
FORECAST_SOURCE_PREFIXES = ("previous_runs", "live_forecast")
FUTURE_KNOWN_COLUMNS = {"target_date", "item_id", "unit_price", "promo_flag"}


class FeatureBuilder:
    """Build training and future features without learning from future rows."""

    def __init__(
        self,
        *,
        calendar: pd.DataFrame,
        weather: pd.DataFrame,
        include_weather: bool = True,
    ) -> None:
        self.calendar = calendar.copy()
        self.weather = weather.copy()
        self.include_weather = include_weather
        self.category_encoders_: dict[str, dict[str, int]] = {}
        self.numeric_scalers_: dict[str, dict[str, float]] = {}
        self._fitted = False

    def fit(self, history: pd.DataFrame) -> FeatureBuilder:
        """Learn JSON-serializable encoders and scaler statistics from history only."""
        self._require_columns(history, {"date", "item_id", "quantity"}, "history")
        categories: dict[str, dict[str, int]] = {}
        scalers: dict[str, dict[str, float]] = {}
        for column in history.columns:
            series = history[column]
            if column == "date":
                continue
            if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
                values = sorted(str(value) for value in series.dropna().unique())
                categories[column] = {value: index for index, value in enumerate(values)}
                continue
            numeric = pd.to_numeric(series, errors="raise").dropna()
            if numeric.empty:
                continue
            mean = float(numeric.mean())
            scale = float(numeric.std(ddof=0))
            scalers[column] = {"mean": mean, "scale": scale if scale else 1.0}
        self.category_encoders_ = categories
        self.numeric_scalers_ = scalers
        self._fitted = True
        return self

    def state_dict(self) -> dict[str, dict[str, dict[str, int | float]]]:
        """Return learned state composed only of JSON-compatible primitives."""
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fit before serializing state")
        return {
            "category_encoders": {
                column: dict(mapping) for column, mapping in self.category_encoders_.items()
            },
            "numeric_scalers": {
                column: dict(parameters) for column, parameters in self.numeric_scalers_.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        calendar: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> FeatureBuilder:
        """Restore a builder from a previously serialized state dictionary."""
        builder = cls(calendar=calendar, weather=weather)
        builder.category_encoders_ = {
            str(column): {str(value): int(code) for value, code in mapping.items()}
            for column, mapping in state["category_encoders"].items()
        }
        builder.numeric_scalers_ = {
            str(column): {
                "mean": float(parameters["mean"]),
                "scale": float(parameters["scale"]),
            }
            for column, parameters in state["numeric_scalers"].items()
        }
        builder._fitted = True
        return builder

    def build(
        self,
        history: pd.DataFrame,
        future_known: pd.DataFrame,
        origin: pd.Timestamp,
        *,
        fit_history: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create as-of features while fitting state only on ``fit_history`` when given."""
        origin = pd.Timestamp(origin)
        train = self._training_features(history, origin)
        train = self._join_calendar(train)
        self.fit(history if fit_history is None else fit_history)
        future = self._future_features(future_known, origin)
        future = self._join_calendar(future)
        if self.include_weather:
            future = future.merge(
                self._weather_at_origin(origin, pd.DatetimeIndex(future["target_date"])),
                on="target_date",
                how="left",
                validate="many_to_one",
            )
        return train, future

    @classmethod
    def _training_features(cls, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        cls._require_columns(history, {"date", "item_id", "quantity"}, "history")
        train = history.copy()
        train["date"] = pd.to_datetime(train["date"], errors="raise").dt.normalize()
        origin_day = cls._local_day(origin)
        if train["date"].gt(origin_day).any():
            raise ValueError("history must not contain rows after origin")
        if train.duplicated(["item_id", "date"]).any():
            raise ValueError("history must have one row per item and date")
        train = train.sort_values(["item_id", "date"]).reset_index(drop=True)
        quantities = train.groupby("item_id", sort=False)["quantity"]
        shifted = quantities.shift(1)
        for lag in LAGS:
            train[f"lag_{lag}"] = quantities.shift(lag)
        shifted_by_item = shifted.groupby(train["item_id"], sort=False)
        for window in ROLLING_WINDOWS:
            rolling = shifted_by_item.rolling(window=window, min_periods=1)
            train[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            train[f"rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
            train[f"rolling_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
            train[f"rolling_max_{window}"] = rolling.max().reset_index(level=0, drop=True)
        train["target_date"] = train["date"]
        return train

    @classmethod
    def _future_features(cls, future_known: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        if "quantity" in future_known.columns:
            raise ValueError("future_known must not contain quantity")
        cls._require_columns(
            future_known,
            FUTURE_KNOWN_COLUMNS,
            "future_known",
        )
        unexpected = sorted(set(future_known.columns).difference(FUTURE_KNOWN_COLUMNS))
        if unexpected:
            raise ValueError(
                "future_known may only contain target_date, item_id, unit_price, and "
                f"promo_flag; unexpected columns: {unexpected}"
            )
        known = future_known.copy()
        known["target_date"] = pd.to_datetime(
            known["target_date"], errors="raise"
        ).dt.normalize()
        expected_dates = pd.date_range(
            cls._local_day(origin) + pd.Timedelta(1, unit="days"), periods=7
        )
        if set(known["target_date"]) != set(expected_dates):
            raise ValueError("future_known must contain exact seven target dates")
        for _, item_rows in known.groupby("item_id", dropna=False):
            if set(item_rows["target_date"]) != set(expected_dates):
                raise ValueError("each item must contain exact seven target dates")
        if known.duplicated(["item_id", "target_date"]).any():
            raise ValueError("future_known has conflicting covariates for an item and target date")
        return known.sort_values(["item_id", "target_date"]).reset_index(drop=True)

    def _join_calendar(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(self.calendar, {"date", *CALENDAR_COLUMNS}, "calendar")
        calendar = self.calendar.loc[:, ["date", *CALENDAR_COLUMNS]].copy()
        calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.normalize()
        if calendar.duplicated("date").any():
            raise ValueError("calendar must have one row per date")
        calendar = calendar.rename(columns={"date": "target_date"})
        base = frame.drop(columns=[column for column in CALENDAR_COLUMNS if column in frame])
        joined = base.merge(
            calendar,
            on="target_date",
            how="left",
            validate="many_to_one",
            indicator="_calendar_join",
        )
        if joined["_calendar_join"].ne("both").any():
            raise ValueError("calendar is missing a required target date")
        return joined.drop(columns="_calendar_join")

    def _weather_at_origin(
        self, origin: pd.Timestamp, target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        missing = sorted(WEATHER_REQUIRED_COLUMNS.difference(self.weather.columns))
        if missing:
            raise WeatherUnavailableError(f"future weather missing columns: {missing}")
        weather = self.weather.copy()
        try:
            target_utc = pd.to_datetime(
                weather["target_date"], errors="raise", utc=True, format="mixed"
            )
            weather["target_date"] = (
                target_utc.dt.tz_convert("Asia/Ho_Chi_Minh")
                .dt.tz_localize(None)
                .dt.normalize()
            )
            weather["_issued_utc"] = pd.to_datetime(
                weather["issued_at"], errors="raise", utc=True, format="mixed"
            )
        except (TypeError, ValueError) as error:
            raise WeatherUnavailableError(f"invalid weather timestamp: {error}") from error

        source = weather["source"].astype("string").str.lower()
        forecast_source = source.fillna("").str.startswith(FORECAST_SOURCE_PREFIXES)
        required_dates = pd.DatetimeIndex(target_dates).normalize().unique()
        eligible = weather.loc[
            forecast_source
            & weather["_issued_utc"].le(self._origin_utc(origin))
            & weather["target_date"].isin(required_dates)
        ].copy()
        if set(eligible["target_date"]) != set(required_dates):
            if not forecast_source.any():
                raise WeatherUnavailableError(
                    "weather must use a genuine previous-runs or live-forecast source"
                )
            raise WeatherUnavailableError(
                "no forecast weather issue at or before origin for every target date"
            )
        selected = (
            eligible.sort_values("_issued_utc")
            .drop_duplicates("target_date", keep="last")
            .loc[
                :,
                ["target_date", "issued_at", *WEATHER_NUMERIC_COLUMNS, "source"],
            ]
        )
        try:
            selected.loc[:, WEATHER_NUMERIC_COLUMNS] = selected.loc[
                :, WEATHER_NUMERIC_COLUMNS
            ].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise WeatherUnavailableError(
                "weather values must be numeric and non-null"
            ) from error
        numeric = selected.loc[:, WEATHER_NUMERIC_COLUMNS]
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype="float64")).all():
            raise WeatherUnavailableError("weather values must be numeric and non-null")
        return selected

    @staticmethod
    def _local_day(value: pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
        return timestamp.normalize()

    @staticmethod
    def _origin_utc(origin: pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(origin)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Ho_Chi_Minh")
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _require_columns(frame: pd.DataFrame, required: set[str], table: str) -> None:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{table} missing columns: {missing}")
