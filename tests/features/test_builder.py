import json

import pandas as pd
import pytest

from fnb_forecast.exceptions import WeatherUnavailableError
from fnb_forecast.features.builder import FeatureBuilder


def make_single_item_sales(quantities: list[int]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(quantities), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "item_id": "coffee",
            "quantity": quantities,
            "unit_price": [30_000.0] * len(quantities),
            "promo_flag": pd.Series([False] * len(quantities), dtype="boolean"),
            "stockout_flag": pd.Series([False] * len(quantities), dtype="boolean"),
            "store_open": pd.Series([True] * len(quantities), dtype="boolean"),
        }
    )


def make_known_days(origin_day: int = 7) -> pd.DataFrame:
    origin = pd.Timestamp("2024-01-01") + pd.Timedelta(origin_day - 1, unit="days")
    targets = pd.date_range(
        origin + pd.Timedelta(1, unit="days"), periods=7, freq="D"
    )
    return pd.DataFrame(
        {
            "target_date": targets,
            "item_id": "coffee",
            "unit_price": [31_000.0 + offset for offset in range(7)],
            "promo_flag": pd.Series(
                [False, True, False, False, True, False, False], dtype="boolean"
            ),
        }
    )


def make_weather(
    origin_day: int = 7,
    *,
    issued_at: pd.Timestamp | None = None,
    source: str = "previous_runs",
) -> pd.DataFrame:
    origin = pd.Timestamp("2024-01-01") + pd.Timedelta(origin_day - 1, unit="days")
    targets = pd.date_range(
        origin + pd.Timedelta(1, unit="days"), periods=7, freq="D"
    )
    issue_origin = origin.tz_localize("Asia/Ho_Chi_Minh")
    return pd.DataFrame(
        {
            "target_date": targets,
            "issued_at": issued_at or issue_origin - pd.Timedelta(1, unit="hours"),
            "temperature": [30.0 + offset for offset in range(7)],
            "rain": [0.0] * 7,
            "humidity": [70.0] * 7,
            "source": source,
        }
    )


def make_calendar() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "weekday": dates.weekday,
            "is_weekend": pd.Series([False] * 14, dtype="boolean"),
            "holiday_name": [None] * 14,
            "days_to_tet": list(range(30, 16, -1)),
            "days_after_tet": list(range(-30, -16)),
            "store_open": pd.Series([True] * 14, dtype="boolean"),
        }
    )


def make_builder(
    *,
    calendar: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> FeatureBuilder:
    return FeatureBuilder(
        calendar=make_calendar() if calendar is None else calendar,
        weather=make_weather() if weather is None else weather,
    )


def test_lag_and_rolling_features_use_only_prior_days() -> None:
    sales = make_single_item_sales([1, 2, 3, 4, 5, 6, 7, 100])

    train, _ = make_builder().build(
        sales.iloc[:7], make_known_days(7), pd.Timestamp("2024-01-07")
    )

    last = train.iloc[-1]
    assert last["lag_1"] == 6
    assert last["rolling_mean_7"] == 3.5
    assert 100 not in train.select_dtypes("number").to_numpy()


def test_build_accepts_price_and_promotion_only_for_exact_seven_target_days() -> None:
    origin = pd.Timestamp("2024-01-07")
    known = make_known_days(7)

    _, future = make_builder().build(make_single_item_sales(list(range(1, 8))), known, origin)

    assert future["target_date"].tolist() == list(pd.date_range("2024-01-08", periods=7))
    assert future["unit_price"].tolist() == [31_000.0 + offset for offset in range(7)]
    assert future["promo_flag"].tolist() == [False, True, False, False, True, False, False]
    assert "quantity" not in future.columns


@pytest.mark.parametrize("bad_date", ["2024-01-07", "2024-01-15"])
def test_build_rejects_known_rows_outside_the_seven_target_days(bad_date: str) -> None:
    known = make_known_days(7)
    known.loc[0, "target_date"] = pd.Timestamp(bad_date)

    with pytest.raises(ValueError, match="exact seven target dates"):
        make_builder().build(
            make_single_item_sales(list(range(1, 8))), known, pd.Timestamp("2024-01-07")
        )


def test_build_rejects_target_quantity_in_future_known() -> None:
    known = make_known_days(7)
    known["quantity"] = 999

    with pytest.raises(ValueError, match="must not contain quantity"):
        make_builder().build(
            make_single_item_sales(list(range(1, 8))), known, pd.Timestamp("2024-01-07")
        )


@pytest.mark.parametrize("extra_column", ["weekday", "temperature", "source", "notes"])
def test_build_rejects_columns_outside_explicit_future_known_schema(
    extra_column: str,
) -> None:
    known = make_known_days(7)
    known[extra_column] = "caller-supplied"

    with pytest.raises(ValueError, match="only contain"):
        make_builder().build(
            make_single_item_sales(list(range(1, 8))),
            known,
            pd.Timestamp("2024-01-07"),
        )


def test_weather_issued_one_minute_after_origin_is_unavailable() -> None:
    origin = pd.Timestamp("2024-01-07 12:00:00", tz="Asia/Ho_Chi_Minh")
    weather = make_weather(
        7, issued_at=origin + pd.Timedelta(1, unit="minutes")
    )

    with pytest.raises(WeatherUnavailableError):
        make_builder(weather=weather).build(
            make_single_item_sales(list(range(1, 8))), make_known_days(7), origin
        )


def test_missing_future_weather_raises_typed_error_instead_of_using_observations() -> None:
    weather = make_weather(7).drop(
        columns=["issued_at", "temperature", "rain", "humidity", "source"]
    )

    with pytest.raises(WeatherUnavailableError):
        make_builder(weather=weather).build(
            make_single_item_sales(list(range(1, 8))),
            make_known_days(7),
            pd.Timestamp("2024-01-07"),
        )


def test_latest_weather_issue_at_or_before_origin_is_joined_by_target_date() -> None:
    origin = pd.Timestamp("2024-01-07 12:00:00", tz="Asia/Ho_Chi_Minh")
    valid = make_weather(7, issued_at=origin - pd.Timedelta(2, unit="hours"))
    valid["temperature"] = [20.0 + offset for offset in range(7)]
    late = valid.copy()
    late["issued_at"] = origin + pd.Timedelta(1, unit="minutes")
    late["temperature"] = 999.0

    _, future = make_builder(weather=pd.concat([late, valid], ignore_index=True)).build(
        make_single_item_sales(list(range(1, 8))),
        make_known_days(7),
        origin,
    )

    assert future["temperature"].tolist() == [20.0 + offset for offset in range(7)]
    assert 999.0 not in future["temperature"].to_numpy()


def test_fit_state_uses_history_only_and_is_json_serializable() -> None:
    history = make_single_item_sales(list(range(1, 8)))
    known = make_known_days(7)
    known["unit_price"] = 9_999_999.0
    builder = make_builder()

    builder.build(history, known, pd.Timestamp("2024-01-07"))
    state = builder.state_dict()

    assert state["numeric_scalers"]["unit_price"]["mean"] == 30_000.0
    assert "9_999_999" not in json.dumps(state)
    json.dumps(state)


def test_calendar_is_joined_by_target_date_for_train_and_future_rows() -> None:
    calendar = make_calendar().sample(frac=1, random_state=4).reset_index(drop=True)

    train, future = make_builder(calendar=calendar).build(
        make_single_item_sales(list(range(1, 8))),
        make_known_days(7),
        pd.Timestamp("2024-01-07"),
    )

    assert train["weekday"].tolist() == list(range(7))
    assert future["weekday"].tolist() == list(range(7))
    assert future["days_to_tet"].tolist() == list(range(23, 16, -1))


def test_calendar_missing_a_required_field_is_rejected() -> None:
    calendar = make_calendar().drop(columns="weekday")

    with pytest.raises(ValueError, match="calendar missing columns"):
        make_builder(calendar=calendar).build(
            make_single_item_sales(list(range(1, 8))),
            make_known_days(7),
            pd.Timestamp("2024-01-07"),
        )


def test_observed_weather_is_rejected_even_when_issued_before_origin() -> None:
    weather = make_weather(source="historical_observation")

    with pytest.raises(WeatherUnavailableError, match="forecast source"):
        make_builder(weather=weather).build(
            make_single_item_sales(list(range(1, 8))),
            make_known_days(7),
            pd.Timestamp("2024-01-07"),
        )


@pytest.mark.parametrize("column", ["issued_at", "target_date"])
def test_malformed_weather_timestamps_raise_typed_error(column: str) -> None:
    weather = make_weather()
    weather[column] = weather[column].astype("object")
    weather.loc[0, column] = "not-a-timestamp"

    with pytest.raises(WeatherUnavailableError, match="timestamp"):
        make_builder(weather=weather).build(
            make_single_item_sales(list(range(1, 8))),
            make_known_days(7),
            pd.Timestamp("2024-01-07"),
        )


def test_missing_target_weather_raises_typed_error() -> None:
    weather = make_weather().iloc[:-1]

    with pytest.raises(WeatherUnavailableError, match="every target date"):
        make_builder(weather=weather).build(
            make_single_item_sales(list(range(1, 8))),
            make_known_days(7),
            pd.Timestamp("2024-01-07"),
        )


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("temperature", None), ("rain", "wet"), ("humidity", None)],
)
def test_invalid_required_weather_values_raise_typed_error(
    column: str, bad_value: object
) -> None:
    weather = make_weather()
    weather[column] = weather[column].astype("object")
    weather.loc[0, column] = bad_value

    with pytest.raises(WeatherUnavailableError, match="numeric and non-null"):
        make_builder(weather=weather).build(
            make_single_item_sales(list(range(1, 8))),
            make_known_days(7),
            pd.Timestamp("2024-01-07"),
        )
