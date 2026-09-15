"""Vectorized revenue and recipe conversion for seven-day forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fnb_forecast.exceptions import DataValidationError


@dataclass
class IngredientNeedResult:
    totals: pd.DataFrame
    detail: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlanningOutput:
    item_forecasts: pd.DataFrame
    revenue_forecast: pd.DataFrame
    ingredient_forecast: pd.DataFrame
    ingredient_detail: pd.DataFrame
    order_proposals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def calculate_revenue(forecasts: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join planned prices one-to-one and calculate item/date revenue."""
    _require_columns(forecasts, {"target_date", "item_id", "yhat"}, "forecasts")
    _require_columns(prices, {"target_date", "item_id", "unit_price"}, "prices")
    forecast = forecasts.copy()
    price_frame = prices.copy()
    for frame in (forecast, price_frame):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise").dt.normalize()
    _reject_duplicate_keys(price_frame, ["target_date", "item_id"], "prices")
    _reject_nonnegative(price_frame, "unit_price", "prices")
    merged = forecast.merge(
        price_frame.loc[:, ["target_date", "item_id", "unit_price"]],
        on=["target_date", "item_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        rows = merged.index[merged["_merge"].ne("both")].tolist()
        raise DataValidationError("prices", ["target_date", "item_id"], rows, "missing price")
    merged["forecast_revenue"] = merged["yhat"] * merged["unit_price"]
    return merged.drop(columns="_merge")


def calculate_ingredient_need(
    forecasts: pd.DataFrame, recipes: pd.DataFrame
) -> IngredientNeedResult:
    """Convert item demand through BOM amounts and waste, preserving audit detail."""
    _require_columns(forecasts, {"target_date", "item_id", "yhat"}, "forecasts")
    _require_columns(
        recipes, {"item_id", "ingredient_id", "amount", "unit", "waste_rate"}, "recipes"
    )
    forecast = forecasts.copy()
    bom = recipes.copy()
    forecast["target_date"] = pd.to_datetime(forecast["target_date"], errors="raise").dt.normalize()
    _reject_duplicate_keys(bom, ["item_id", "ingredient_id"], "recipes")
    _reject_nonnegative(bom, "amount", "recipes")
    waste = pd.to_numeric(bom["waste_rate"], errors="coerce")
    if waste.isna().any() or waste.lt(0).any() or waste.gt(1).any():
        rows = waste.index[waste.isna() | waste.lt(0) | waste.gt(1)].tolist()
        raise DataValidationError(
            "recipes", ["waste_rate"], rows, "waste_rate must be between 0 and 1"
        )
    units = bom.groupby("ingredient_id", dropna=False)["unit"].nunique()
    mixed = units[units.gt(1)]
    if not mixed.empty:
        rows = bom.index[bom["ingredient_id"].isin(mixed.index)].tolist()
        raise DataValidationError("recipes", ["ingredient_id", "unit"], rows, "mixed units")
    forecast_items = set(forecast["item_id"].dropna().unique())
    recipe_items = set(bom["item_id"].dropna().unique())
    missing_items = sorted(forecast_items.difference(recipe_items), key=str)
    warnings = [f"Missing BOM for item: {item}" for item in missing_items]
    joined = forecast.merge(
        bom,
        on="item_id",
        how="inner",
        validate="many_to_many",
    )
    joined["forecast_need"] = joined["yhat"] * joined["amount"] * (1.0 + joined["waste_rate"])
    detail_columns = [
        "target_date",
        "ingredient_id",
        "unit",
        "item_id",
        "forecast_need",
    ]
    detail = joined.loc[:, detail_columns].rename(columns={"item_id": "source_item_id"})
    totals = (
        detail.groupby(["target_date", "ingredient_id", "unit"], as_index=False, dropna=False)[
            "forecast_need"
        ]
        .sum()
        .sort_values(["target_date", "ingredient_id"])
        .reset_index(drop=True)
    )
    return IngredientNeedResult(totals=totals, detail=detail, warnings=warnings)


def _require_columns(frame: pd.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(table, missing, [], "missing columns")


def _reject_duplicate_keys(frame: pd.DataFrame, keys: list[str], table: str) -> None:
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        raise DataValidationError(table, keys, frame.index[duplicate].tolist(), "duplicate keys")


def _reject_nonnegative(frame: pd.DataFrame, column: str, table: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() | values.lt(0)
    if invalid.any():
        raise DataValidationError(
            table, [column], frame.index[invalid].tolist(), "must be non-negative"
        )
