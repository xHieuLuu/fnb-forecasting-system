"""Validation-derived safety stock and FEFO order proposals."""

from __future__ import annotations

import math

import pandas as pd

from fnb_forecast.exceptions import DataValidationError


def build_ingredient_residuals(
    validation_predictions: pd.DataFrame, recipes: pd.DataFrame
) -> pd.DataFrame:
    """Convert validation actuals and forecasts through the same recipe table."""
    required = {"target_date", "item_id", "actual", "yhat", "role"}
    missing = sorted(required.difference(validation_predictions.columns))
    if missing:
        raise ValueError(f"validation predictions missing columns: {missing}")
    roles = set(validation_predictions["role"].dropna().astype(str))
    if roles != {"validation"}:
        raise ValueError("ingredient residuals require validation rows only")
    recipe_required = {"item_id", "ingredient_id", "amount", "unit", "waste_rate"}
    missing = sorted(recipe_required.difference(recipes.columns))
    if missing:
        raise DataValidationError("recipes", missing, [], "missing columns")
    rows = validation_predictions.merge(recipes, on="item_id", how="inner", validate="many_to_many")
    rows["actual_need"] = rows["actual"] * rows["amount"] * (1.0 + rows["waste_rate"])
    rows["forecast_need"] = rows["yhat"] * rows["amount"] * (1.0 + rows["waste_rate"])
    return (
        rows.groupby(["target_date", "ingredient_id", "unit"], as_index=False)[
            ["actual_need", "forecast_need"]
        ]
        .sum()
        .sort_values(["target_date", "ingredient_id"])
        .reset_index(drop=True)
        .assign(role="validation")
    )


def compute_safety_stock(
    residuals: pd.DataFrame,
    ingredient_master: pd.DataFrame,
    service_level: float,
) -> pd.DataFrame:
    """Compute empirical under-forecast quantiles over each protection window."""
    if service_level not in {0.90, 0.95, 0.98}:
        raise ValueError("service_level must be one of 0.90, 0.95, or 0.98")
    required = {"role", "ingredient_id", "target_date", "actual_need", "forecast_need"}
    missing = sorted(required.difference(residuals.columns))
    if missing:
        raise ValueError(f"residuals missing columns: {missing}")
    roles = set(residuals["role"].dropna().astype(str))
    if roles != {"validation"}:
        raise ValueError("safety stock requires validation residuals only")
    master_required = {"ingredient_id", "lead_time_days", "review_period_days"}
    missing = sorted(master_required.difference(ingredient_master.columns))
    if missing:
        raise DataValidationError("ingredient_master", missing, [], "missing columns")
    data = residuals.copy()
    data["target_date"] = pd.to_datetime(data["target_date"], errors="raise").dt.normalize()
    data["underforecast"] = (data["actual_need"] - data["forecast_need"]).clip(lower=0.0)
    master = ingredient_master.drop_duplicates("ingredient_id").copy()
    result_rows: list[dict[str, object]] = []
    quantile = service_level
    for ingredient_id, values in data.groupby("ingredient_id", sort=True, dropna=False):
        parameters = master.loc[master["ingredient_id"].eq(ingredient_id)].iloc[0]
        coverage = int(parameters["lead_time_days"]) + int(parameters["review_period_days"])
        values = values.sort_values("target_date").copy()
        rolling = values["underforecast"].rolling(coverage, min_periods=coverage).sum().dropna()
        result_rows.append(
            {
                "ingredient_id": ingredient_id,
                "safety_stock": float(rolling.quantile(quantile)),
                "sample_count": int(len(rolling)),
                "service_level": service_level,
                "coverage_days": coverage,
            }
        )
    return pd.DataFrame(result_rows)


def suggest_orders(
    ingredient_need: pd.DataFrame,
    safety_stock: pd.DataFrame,
    inventory_lots: pd.DataFrame,
    scheduled_receipts: pd.DataFrame,
    ingredient_master: pd.DataFrame,
    *,
    order_date: pd.Timestamp,
) -> pd.DataFrame:
    """Create FEFO-aware order proposals with pack rounding and perishability warnings."""
    need_required = {"target_date", "ingredient_id", "forecast_need"}
    missing = sorted(need_required.difference(ingredient_need.columns))
    if missing:
        raise DataValidationError("ingredient_need", missing, [], "missing columns")
    order_day = pd.Timestamp(order_date).normalize()
    master = ingredient_master.set_index("ingredient_id", drop=False)
    safety = safety_stock.set_index("ingredient_id")["safety_stock"]
    lots = inventory_lots.copy()
    if not lots.empty:
        lots["expiry_date"] = pd.to_datetime(lots["expiry_date"], errors="raise").dt.normalize()
        lots["received_date"] = pd.to_datetime(lots["received_date"], errors="raise").dt.normalize()
    receipts = scheduled_receipts.copy()
    if not receipts.empty:
        receipts["arrival_date"] = pd.to_datetime(
            receipts["arrival_date"], errors="raise"
        ).dt.normalize()
    rows: list[dict[str, object]] = []
    for ingredient_id, values in ingredient_need.groupby("ingredient_id", sort=True, dropna=False):
        if ingredient_id not in master.index:
            continue
        parameters = master.loc[ingredient_id]
        lead = int(parameters["lead_time_days"])
        review = int(parameters["review_period_days"])
        coverage_need = float(
            values.loc[
                pd.to_datetime(values["target_date"]).between(
                    order_day,
                    order_day + pd.Timedelta(lead + review - 1, unit="D"),
                ),
                "forecast_need",
            ].sum()
        )
        usable = allocate_fefo(
            lots.loc[lots["ingredient_id"].eq(ingredient_id)] if not lots.empty else lots,
            coverage_need_date=order_day + pd.Timedelta(lead + review - 1, unit="D"),
        )
        scheduled = (
            float(
                receipts.loc[
                    receipts["ingredient_id"].eq(ingredient_id)
                    & receipts["arrival_date"].le(order_day + pd.Timedelta(lead + review, unit="D")),
                    "quantity",
                ].sum()
            )
            if not receipts.empty
            else 0.0
        )
        safety_value = float(safety.get(ingredient_id, 0.0))
        target_stock = coverage_need + safety_value
        raw_order = max(0.0, target_stock - usable - scheduled)
        pack_size = float(parameters["pack_size"])
        suggested = math.ceil(raw_order / pack_size) * pack_size if raw_order > 0 else 0.0
        perishable = bool(parameters.get("perishable", False))
        shelf_life = parameters.get("shelf_life_days")
        warning = ""
        cap_applied = False
        waste_risk = False
        if perishable and pd.notna(shelf_life):
            cap_applied = True
            cap_need = (
                float(
                    values.loc[
                        pd.to_datetime(values["target_date"]).between(
                            order_day,
                            order_day + pd.Timedelta(int(shelf_life), unit="D"),
                        ),
                        "forecast_need",
                    ].sum()
                )
                + safety_value
            )
            if suggested > max(0.0, cap_need - usable - scheduled):
                waste_risk = suggested > 0
                warning = "Pack size may exceed shelf-life demand" if waste_risk else ""
        elif perishable:
            warning = "Missing shelf life for perishable ingredient"
        rows.append(
            {
                "ingredient_id": ingredient_id,
                "coverage_forecast_need": coverage_need,
                "safety_stock": safety_value,
                "usable_on_hand": usable,
                "scheduled_receipts": scheduled,
                "target_stock": target_stock,
                "raw_order": raw_order,
                "suggested_order": suggested,
                "shelf_life_cap_applied": cap_applied,
                "waste_risk_warning": waste_risk,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def allocate_fefo(lots: pd.DataFrame, *, coverage_need_date: pd.Timestamp) -> float:
    """Return stock usable through the coverage date, ordered FEFO."""
    if lots.empty:
        return 0.0
    expiry = pd.to_datetime(lots["expiry_date"], errors="raise").dt.normalize()
    usable = lots.loc[expiry.ge(pd.Timestamp(coverage_need_date).normalize())].copy()
    return float(usable["on_hand"].sum())
