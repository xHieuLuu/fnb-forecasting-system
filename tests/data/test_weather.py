import httpx
import pandas as pd
import pytest

from fnb_forecast.data.weather import OpenMeteoWeatherClient, assert_weather_known_at_origin
from fnb_forecast.exceptions import WeatherUnavailableError


def test_future_issued_weather_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "issued_at": ["2025-01-03"],
            "target_date": ["2025-01-04"],
            "temperature": [31.0],
            "rain": [4.0],
            "humidity": [80.0],
            "source": ["historical_forecast"],
        }
    )

    with pytest.raises(WeatherUnavailableError):
        assert_weather_known_at_origin(frame, pd.Timestamp("2025-01-02"))


def test_previous_runs_uses_cached_response_when_service_is_unavailable(tmp_path) -> None:
    payload = {
        "hourly": {
            "time": ["2025-01-04T00:00", "2025-01-04T12:00"],
            "temperature_2m_previous_day2": [30.0, 32.0],
            "relative_humidity_2m_previous_day2": [70.0, 80.0],
            "rain_previous_day2": [1.0, 3.0],
        }
    }

    client = OpenMeteoWeatherClient(
        latitude=10.8231,
        longitude=106.6297,
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )
    targets = pd.DatetimeIndex(["2025-01-04"])
    first = client.fetch_previous_runs_forecast(pd.Timestamp("2025-01-02"), targets)

    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={}))
    )
    cached = client.fetch_previous_runs_forecast(pd.Timestamp("2025-01-02"), targets)

    assert first.loc[0, "temperature"] == 31.0
    assert first.loc[0, "rain"] == 4.0
    assert cached.loc[0, "source"] == "previous_runs_cache"


def test_historical_observations_are_daily_targets_for_the_synthetic_generator(tmp_path) -> None:
    payload = {
        "hourly": {
            "time": ["2025-01-04T00:00", "2025-01-04T12:00"],
            "temperature_2m": [30.0, 32.0],
            "relative_humidity_2m": [70.0, 80.0],
            "rain": [1.0, 3.0],
        }
    }
    client = OpenMeteoWeatherClient(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )

    observed = client.fetch_historical_observations(
        pd.Timestamp("2025-01-04").date(), pd.Timestamp("2025-01-04").date()
    )

    assert observed.columns.tolist() == ["date", "temperature", "rain", "humidity", "source"]
    assert observed.loc[0, "temperature"] == 31.0
    assert observed.loc[0, "rain"] == 4.0
    assert observed.loc[0, "source"] == "historical_observation"


def test_live_forecast_rejects_a_response_acquired_after_a_historical_origin(tmp_path) -> None:
    payload = {
        "hourly": {
            "time": ["2025-01-04T00:00"],
            "temperature_2m": [30.0],
            "relative_humidity_2m": [70.0],
            "rain": [1.0],
        }
    }
    client = OpenMeteoWeatherClient(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )

    with pytest.raises(WeatherUnavailableError):
        client.fetch_live_forecast(pd.Timestamp("2020-01-01"), pd.DatetimeIndex(["2025-01-04"]))


@pytest.mark.parametrize("cache_contents", ["[]", "{not-json"])
def test_invalid_cached_weather_raises_domain_error_on_provider_failure(
    tmp_path, cache_contents
) -> None:
    payload = {
        "hourly": {
            "time": ["2025-01-04T00:00"],
            "temperature_2m_previous_day2": [30.0],
            "relative_humidity_2m_previous_day2": [70.0],
            "rain_previous_day2": [1.0],
        }
    }
    client = OpenMeteoWeatherClient(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )
    targets = pd.DatetimeIndex(["2025-01-04"])
    client.fetch_previous_runs_forecast(pd.Timestamp("2025-01-02"), targets)
    cache_file = next((tmp_path / "weather").glob("*.json"))
    cache_file.write_text(cache_contents, encoding="utf-8")
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={}))
    )

    with pytest.raises(WeatherUnavailableError):
        client.fetch_previous_runs_forecast(pd.Timestamp("2025-01-02"), targets)


def test_unreadable_cached_weather_raises_domain_error_on_provider_failure(
    tmp_path, monkeypatch
) -> None:
    payload = {
        "hourly": {
            "time": ["2025-01-04T00:00"],
            "temperature_2m_previous_day2": [30.0],
            "relative_humidity_2m_previous_day2": [70.0],
            "rain_previous_day2": [1.0],
        }
    }
    client = OpenMeteoWeatherClient(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )
    targets = pd.DatetimeIndex(["2025-01-04"])
    client.fetch_previous_runs_forecast(pd.Timestamp("2025-01-02"), targets)
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={}))
    )

    def unreadable_cache(self, *args, **kwargs):
        raise OSError("cache unavailable")

    monkeypatch.setattr(
        type(next((tmp_path / "weather").glob("*.json"))), "read_text", unreadable_cache
    )
    with pytest.raises(WeatherUnavailableError):
        client.fetch_previous_runs_forecast(pd.Timestamp("2025-01-02"), targets)


def test_previous_runs_accepts_timezone_aware_origin_with_naive_target_dates(tmp_path) -> None:
    payload = {
        "hourly": {
            "time": ["2025-01-04T00:00"],
            "temperature_2m_previous_day2": [30.0],
            "relative_humidity_2m_previous_day2": [70.0],
            "rain_previous_day2": [1.0],
        }
    }
    client = OpenMeteoWeatherClient(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )

    result = client.fetch_previous_runs_forecast(
        pd.Timestamp("2025-01-02", tz="Asia/Ho_Chi_Minh"),
        pd.DatetimeIndex(["2025-01-04"]),
    )

    assert result.loc[0, "target_date"] == pd.Timestamp("2025-01-04")
    assert result.loc[0, "source"] == "previous_runs"
