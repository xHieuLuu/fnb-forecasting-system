"""Vietnam holiday and calendar features."""

import unicodedata
from datetime import date

import holidays
import pandas as pd

from fnb_forecast.exceptions import DataValidationError


def _normalized_label(label: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", label).lower()
        if not unicodedata.combining(character)
    )
    return normalized.replace("\u0111", "d")


def _tet_day(year: int) -> date:
    vietnam_holidays = holidays.country_holidays("VN", years=year)
    fallback: date | None = None
    for holiday_day, label in vietnam_holidays.items():
        normalized = _normalized_label(label)
        if normalized.startswith("tet nguyen dan"):
            return holiday_day
        if any(term in normalized for term in ("lunar new year", "vietnamese new year")):
            return holiday_day
        if "tet" in normalized and fallback is None:
            fallback = holiday_day
    if fallback is not None:
        return fallback
    raise ValueError(f"Vietnam holiday calendar has no Tet entry for {year}")


def build_vietnam_calendar(start: date, end: date) -> pd.DataFrame:
    """Build daily Vietnam calendar features for an inclusive date range."""
    if start > end:
        raise DataValidationError("calendar", ["date"], [], "start date must not be after end date")

    dates = pd.date_range(start, end, freq="D")
    holiday_by_date: dict[date, str] = {}
    tet_by_year: dict[int, date] = {}
    for year in range(start.year, end.year + 1):
        vietnam_holidays = holidays.country_holidays("VN", years=year)
        holiday_by_date.update(vietnam_holidays)
        tet_by_year[year] = _tet_day(year)

    calendar = pd.DataFrame({"date": dates})
    calendar["weekday"] = calendar["date"].dt.weekday
    calendar["week_of_year"] = calendar["date"].dt.isocalendar().week.astype("int64")
    calendar["month"] = calendar["date"].dt.month
    calendar["is_weekend"] = calendar["weekday"].ge(5).astype("boolean")
    calendar["holiday_name"] = calendar["date"].dt.date.map(holiday_by_date)
    calendar["is_holiday"] = calendar["holiday_name"].notna().astype("boolean")
    tet_dates = calendar["date"].dt.year.map(tet_by_year)
    calendar["days_to_tet"] = [
        (tet_day - day).days
        for day, tet_day in zip(calendar["date"].dt.date, tet_dates, strict=True)
    ]
    calendar["days_after_tet"] = [
        (day - tet_day).days
        for day, tet_day in zip(calendar["date"].dt.date, tet_dates, strict=True)
    ]
    return calendar
