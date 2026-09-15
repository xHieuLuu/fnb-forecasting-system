"""Leakage-safe Open-Meteo weather retrieval."""

from __future__ import annotations

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from fnb_forecast.exceptions import WeatherUnavailableError

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BASE_VARIABLES = ("temperature_2m", "relative_humidity_2m", "rain")


def assert_weather_known_at_origin(frame: pd.DataFrame, origin: pd.Timestamp) -> None:
    """Reject weather values that were issued after the forecast origin."""
    issued = pd.to_datetime(frame["issued_at"], utc=True)
    origin = pd.Timestamp(origin)
    origin_utc = (
        origin.tz_localize("Asia/Ho_Chi_Minh").tz_convert("UTC")
        if origin.tzinfo is None
        else origin.tz_convert("UTC")
    )
    if issued.gt(origin_utc).any():
        raise WeatherUnavailableError("weather issuance is later than forecast origin")


class OpenMeteoWeatherClient:
    """Open-Meteo client that stores raw HTTP responses in a local disk cache."""

    def __init__(
        self,
        latitude: float = 10.8231,
        longitude: float = 106.6297,
        *,
        cache_root: Path = Path(".cache"),
        client: httpx.Client | None = None,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.cache_root = Path(cache_root) / "weather"
        self._client = client or httpx.Client(timeout=30.0)

    def fetch_historical_observations(self, start: date, end: date) -> pd.DataFrame:
        """Retrieve daily historical observations for synthetic-target generation."""
        payload, source, _ = self._request_json(
            ARCHIVE_URL,
            self._base_params(start, end, BASE_VARIABLES),
            "historical_observation",
        )
        return self._daily_frame(payload, BASE_VARIABLES, source, issued_at=None).rename(
            columns={"target_date": "date"}
        )

    def fetch_previous_runs_forecast(
        self, origin: pd.Timestamp, target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Retrieve only forecasts issued on or before ``origin`` for each target day."""
        normalized_origin = self._local_day(origin)
        frames: list[pd.DataFrame] = []
        for target in self._local_days(target_dates).unique().sort_values():
            horizon = int((target - normalized_origin).days)
            if horizon not in range(1, 8):
                raise WeatherUnavailableError(
                    "previous-runs forecasts require target horizons from 1 to 7 days"
                )
            variables = tuple(f"{name}_previous_day{horizon}" for name in BASE_VARIABLES)
            payload, source, _ = self._request_json(
                PREVIOUS_RUNS_URL,
                self._base_params(target.date(), target.date(), variables),
                "previous_runs",
            )
            frame = self._daily_frame(
                payload,
                variables,
                source,
                issued_at=(target - pd.Timedelta(horizon, unit="D")).tz_localize(
                    "Asia/Ho_Chi_Minh"
                ),
            )
            frames.append(frame)
        result = self._combine_forecasts(frames)
        assert_weather_known_at_origin(result, origin)
        return result

    def fetch_live_forecast(
        self, origin: pd.Timestamp, target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Retrieve live forecasts for future target days and enforce their issuance time."""
        targets = self._local_days(target_dates).unique().sort_values()
        if targets.empty:
            return self._empty_forecast_frame()
        payload, source, acquired_at = self._request_json(
            LIVE_FORECAST_URL,
            self._base_params(targets.min().date(), targets.max().date(), BASE_VARIABLES),
            "live_forecast",
        )
        frame = self._daily_frame(payload, BASE_VARIABLES, source, issued_at=acquired_at)
        result = frame.loc[frame["target_date"].isin(targets)].reset_index(drop=True)
        assert_weather_known_at_origin(result, origin)
        return result

    def _base_params(
        self, start: date, end: date, variables: tuple[str, ...]
    ) -> dict[str, str | float]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(variables),
            "timezone": "Asia/Ho_Chi_Minh",
        }

    def _request_json(
        self, url: str, params: dict[str, str | float], source: str
    ) -> tuple[dict[str, Any], str, pd.Timestamp]:
        request = self._client.build_request("GET", url, params=params)
        cache_path = self.cache_root / f"{sha256(str(request.url).encode()).hexdigest()}.json"
        try:
            response = self._client.send(request)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            if cache_path.exists():
                payload, acquired_at = self._read_cache(cache_path)
                return payload, f"{source}_cache", acquired_at
            raise WeatherUnavailableError(f"weather provider unavailable: {error}") from error
        acquired_at = pd.Timestamp.now(tz="UTC")
        entry = {"acquired_at": acquired_at.isoformat(), "payload": payload}
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(entry), encoding="utf-8")
        except OSError as error:
            raise WeatherUnavailableError(f"weather cache unavailable: {error}") from error
        return payload, source, acquired_at

    @staticmethod
    def _read_cache(cache_path: Path) -> tuple[dict[str, Any], pd.Timestamp]:
        try:
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict):
                raise ValueError("cache entry must be an object")
            payload = entry["payload"]
            acquired_at = pd.Timestamp(pd.to_datetime(entry["acquired_at"], utc=True))
            if not isinstance(payload, dict) or pd.isna(acquired_at):
                raise ValueError("cache entry has an invalid schema")
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise WeatherUnavailableError(
                f"weather cache unavailable or invalid: {error}"
            ) from error
        return payload, acquired_at

    @staticmethod
    def _local_day(value: pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
        return timestamp.normalize()

    @classmethod
    def _local_days(cls, values: pd.DatetimeIndex) -> pd.DatetimeIndex:
        dates = pd.DatetimeIndex(values)
        if dates.tz is not None:
            dates = dates.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
        return dates.normalize()

    @staticmethod
    def _daily_frame(
        payload: dict[str, Any],
        variables: tuple[str, ...],
        source: str,
        issued_at: pd.Timestamp | None,
    ) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise WeatherUnavailableError("weather provider returned no hourly data")
        data = pd.DataFrame({"target_date": pd.to_datetime(hourly["time"])})
        try:
            data["temperature"] = hourly[variables[0]]
            data["humidity"] = hourly[variables[1]]
            data["rain"] = hourly[variables[2]]
        except KeyError as error:
            raise WeatherUnavailableError(
                f"weather provider omitted variable: {error.args[0]}"
            ) from error
        data["target_date"] = data["target_date"].dt.normalize()
        aggregations = {"temperature": "mean", "humidity": "mean", "rain": "sum"}
        result = data.groupby("target_date", as_index=False).agg(aggregations)
        result["source"] = source
        if issued_at is None:
            return result.loc[:, ["target_date", "temperature", "rain", "humidity", "source"]]
        result["issued_at"] = pd.Timestamp(issued_at)
        return result.loc[
            :, ["issued_at", "target_date", "temperature", "rain", "humidity", "source"]
        ]

    @staticmethod
    def _empty_forecast_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["issued_at", "target_date", "temperature", "rain", "humidity", "source"]
        )

    @staticmethod
    def _combine_forecasts(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return OpenMeteoWeatherClient._empty_forecast_frame()
        return (
            pd.concat(frames, ignore_index=True).sort_values("target_date").reset_index(drop=True)
        )
