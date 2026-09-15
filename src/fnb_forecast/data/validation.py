"""Validation and normalization for the project's tabular input bundle."""

from collections.abc import Iterable

import pandas as pd

from fnb_forecast.contracts import DataBundle
from fnb_forecast.exceptions import DataValidationError

REQUIRED_COLUMNS = {
    "daily_sales": {
        "date",
        "item_id",
        "quantity",
        "unit_price",
        "promo_flag",
        "stockout_flag",
        "store_open",
    },
    "item_master": {"item_id", "item_name", "category", "launch_date", "active"},
    "calendar": {
        "date",
        "weekday",
        "is_weekend",
        "holiday_name",
        "days_to_tet",
        "days_after_tet",
        "store_open",
    },
    "weather": {"issued_at", "target_date", "temperature", "rain", "humidity", "source"},
    "recipes": {"item_id", "ingredient_id", "amount", "unit", "waste_rate"},
    "ingredient_master": {
        "ingredient_id",
        "name",
        "base_unit",
        "pack_size",
        "lead_time_days",
        "review_period_days",
        "shelf_life_days",
    },
    "inventory_lots": {"ingredient_id", "lot_id", "on_hand", "expiry_date", "received_date"},
    "scheduled_receipts": {"ingredient_id", "arrival_date", "quantity"},
}

DATE_COLUMNS = {
    "daily_sales": ("date",),
    "item_master": ("launch_date",),
    "calendar": ("date",),
    "weather": ("issued_at", "target_date"),
    "recipes": (),
    "ingredient_master": (),
    "inventory_lots": ("expiry_date", "received_date"),
    "scheduled_receipts": ("arrival_date",),
}

BOOLEAN_COLUMNS = {
    "daily_sales": ("promo_flag", "stockout_flag", "store_open"),
    "item_master": ("active",),
    "calendar": ("is_weekend", "store_open"),
    "weather": (),
    "recipes": (),
    "ingredient_master": (),
    "inventory_lots": (),
    "scheduled_receipts": (),
}


def _require_columns(table: str, frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataValidationError(table, missing, [], "missing columns")


def _normalize_frame(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(table, frame, REQUIRED_COLUMNS[table])
    out = frame.copy()
    for column in DATE_COLUMNS[table]:
        out[column] = pd.to_datetime(out[column], errors="raise").dt.normalize()
    for column in BOOLEAN_COLUMNS[table]:
        out[column] = out[column].astype("boolean")
    return out


def _reject_negative(table: str, frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        bad = frame.index[frame[column].lt(0) | frame[column].isna()].tolist()
        if bad:
            raise DataValidationError(table, [column], bad, "must be non-negative and non-null")


def validate_daily_sales(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate observations keyed by a single item and normalized day."""
    out = _normalize_frame("daily_sales", frame)
    unknown_store_open = out.index[out["store_open"].isna()].tolist()
    if unknown_store_open:
        raise DataValidationError(
            "daily_sales", ["store_open"], unknown_store_open, "must be non-null"
        )
    invalid_quantity = out.index[
        (out["store_open"] & (out["quantity"].lt(0) | out["quantity"].isna()))
        | (~out["store_open"] & out["quantity"].notna())
    ].tolist()
    if invalid_quantity:
        raise DataValidationError(
            "daily_sales",
            ["quantity"],
            invalid_quantity,
            "must be non-negative and non-null when open and null when closed",
        )
    _reject_negative("daily_sales", out, ("unit_price",))
    duplicates = out.index[out.duplicated(["date", "item_id"], keep=False)].tolist()
    if duplicates:
        raise DataValidationError("daily_sales", ["date", "item_id"], duplicates, "duplicate key")
    return out.sort_values(["item_id", "date"]).reset_index(drop=True)


def _validate_table(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_frame(table, frame)
    numeric_columns = {
        "recipes": ("amount", "waste_rate"),
        "ingredient_master": ("pack_size", "lead_time_days", "review_period_days"),
        "inventory_lots": ("on_hand",),
        "scheduled_receipts": ("quantity",),
    }
    if table in numeric_columns:
        _reject_negative(table, out, numeric_columns[table])
    return out


def complete_daily_sales(
    sales: pd.DataFrame, items: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Expand sales to every item-day, zero-filling only open missing days."""
    daily = validate_daily_sales(sales)
    _require_columns("item_master", items, {"item_id"})
    _require_columns("calendar", calendar, {"date", "store_open"})
    days = calendar.loc[:, ["date", "store_open"]].copy()
    days["date"] = pd.to_datetime(days["date"], errors="raise").dt.normalize()
    days["store_open"] = days["store_open"].astype("boolean")
    duplicates = days.index[days.duplicated(["date"], keep=False)].tolist()
    if duplicates:
        raise DataValidationError("calendar", ["date"], duplicates, "duplicate key")
    grid = (
        items.loc[:, ["item_id"]]
        .drop_duplicates()
        .merge(days, how="cross")
    )
    result = grid.merge(
        daily.drop(columns="store_open"), on=["item_id", "date"], how="left", indicator=True
    )
    missing_open = result["_merge"].eq("left_only") & result["store_open"].fillna(False)
    result.loc[missing_open, "quantity"] = 0
    result = result.drop(columns="_merge")
    return result.sort_values(["item_id", "date"]).reset_index(drop=True)


def validate_bundle(bundle: DataBundle) -> DataBundle:
    """Return a schema-validated, date-normalized copy of a data bundle."""
    return DataBundle(
        daily_sales=validate_daily_sales(bundle.daily_sales),
        item_master=_validate_table("item_master", bundle.item_master),
        calendar=_validate_table("calendar", bundle.calendar),
        weather=_validate_table("weather", bundle.weather),
        recipes=_validate_table("recipes", bundle.recipes),
        ingredient_master=_validate_table("ingredient_master", bundle.ingredient_master),
        inventory_lots=_validate_table("inventory_lots", bundle.inventory_lots),
        scheduled_receipts=_validate_table("scheduled_receipts", bundle.scheduled_receipts),
        provenance=bundle.provenance,
    )
