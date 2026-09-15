"""Deterministic Vietnam food-and-beverage scenario generation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, model_validator

from fnb_forecast.config import ProjectConfig, StrictModel
from fnb_forecast.contracts import DataBundle, DataProvenance

from .calendar import build_vietnam_calendar
from .io import canonical_bundle_checksum
from .validation import validate_bundle


class ItemDefinition(StrictModel):
    item_id: str
    item_name: str
    category: str
    base_price: float = Field(gt=0)
    base_log_demand: float
    weekday_effect: dict[int, float]
    month_effect: dict[int, float]
    temperature_beta: float
    rain_beta: float
    holiday_beta: float
    launch_date: date

    @model_validator(mode="after")
    def require_complete_seasonal_effects(self) -> "ItemDefinition":
        if set(self.weekday_effect) != set(range(7)):
            raise ValueError("weekday_effect must define weekdays 0 through 6")
        if set(self.month_effect) != set(range(1, 13)):
            raise ValueError("month_effect must define months 1 through 12")
        return self


class IngredientDefinition(StrictModel):
    ingredient_id: str
    name: str
    base_unit: str
    waste_rate: float = Field(ge=0, lt=1)
    pack_size: float = Field(gt=0)
    lead_time_days: int = Field(ge=0)
    review_period_days: int = Field(gt=0)
    shelf_life_days: int = Field(gt=0)
    perishable: bool


class RecipeDefinition(StrictModel):
    item_id: str
    ingredient_id: str
    amount: float = Field(gt=0)
    unit: str


class SyntheticScenarioDefinition(StrictModel):
    dispersion: float = Field(gt=0)
    promo_beta: float
    items: list[ItemDefinition]
    ingredients: list[IngredientDefinition]
    recipes: list[RecipeDefinition]

    @model_validator(mode="after")
    def require_catalog_integrity(self) -> "SyntheticScenarioDefinition":
        if len(self.items) != 15 or len({item.item_id for item in self.items}) != 15:
            raise ValueError("synthetic scenario must define exactly 15 unique items")
        ingredient_ids = {ingredient.ingredient_id for ingredient in self.ingredients}
        if len(self.ingredients) != 15 or len(ingredient_ids) != 15:
            raise ValueError("synthetic scenario must define exactly 15 unique ingredients")
        item_ids = {item.item_id for item in self.items}
        if any(recipe.item_id not in item_ids for recipe in self.recipes):
            raise ValueError("recipes contain an unknown item_id")
        if any(recipe.ingredient_id not in ingredient_ids for recipe in self.recipes):
            raise ValueError("recipes contain an unknown ingredient_id")
        return self


def load_synthetic_definition(path: Path) -> SyntheticScenarioDefinition:
    """Load a strict synthetic-scenario definition from UTF-8 YAML."""
    return SyntheticScenarioDefinition.model_validate(yaml.safe_load(Path(path).read_text("utf-8")))


def _weather_for_scenario(
    observed_weather: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    required = {"date", "temperature", "rain", "humidity", "source"}
    missing = sorted(required.difference(observed_weather.columns))
    if missing:
        raise ValueError(f"observed_weather missing columns: {missing}")
    observed = observed_weather.loc[:, sorted(required)].copy()
    observed["date"] = pd.to_datetime(observed["date"], errors="raise").dt.normalize()
    expected = pd.date_range(start, end, freq="D")
    if observed["date"].duplicated().any() or set(observed["date"]) != set(expected):
        raise ValueError("observed_weather must contain exactly one row for every scenario day")
    observed = observed.set_index("date").loc[expected]
    observed.index.name = "date"
    observed = observed.reset_index()
    return observed


def _calendar_with_tet_closures(start: date, end: date) -> pd.DataFrame:
    calendar = build_vietnam_calendar(start, end)
    calendar["store_open"] = ~calendar["days_to_tet"].isin([0, 1])
    return calendar


def _sales_frame(
    project: ProjectConfig,
    definition: SyntheticScenarioDefinition,
    calendar: pd.DataFrame,
    observed_weather: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(project.scenario.seed)
    weather = observed_weather.set_index("date")
    final_test_start = pd.Timestamp(project.scenario.end_date) - pd.Timedelta(27, unit="D")
    first_day = pd.Timestamp(project.scenario.start_date)
    records: list[dict[str, object]] = []
    for item_index, item in enumerate(definition.items):
        for day_index, calendar_row in calendar.iterrows():
            day = pd.Timestamp(calendar_row["date"])
            conditions = weather.loc[day]
            promo_flag = (day.dayofyear + item_index * 17) % 31 < 4
            trend = 0.0004 * (day - first_day).days
            preference_drift = 0.08 * np.sin((day_index + item_index * 11) / 45.0)
            random_shock = rng.normal(0.0, 0.08)
            log_mu = (
                item.base_log_demand
                + item.weekday_effect[int(calendar_row["weekday"])]
                + item.month_effect[int(calendar_row["month"])]
                + item.temperature_beta * (float(conditions["temperature"]) - 28.0)
                + item.rain_beta * float(float(conditions["rain"]) > 0)
                + item.holiday_beta * float(bool(calendar_row["is_holiday"]))
                + definition.promo_beta * float(promo_flag)
                + trend
                + preference_drift
                + random_shock
            )
            mu = np.exp(log_mu)
            probability = definition.dispersion / (definition.dispersion + mu)
            latent_quantity = int(rng.negative_binomial(definition.dispersion, probability))
            is_open = bool(calendar_row["store_open"])
            is_active = day.date() >= item.launch_date
            stockout_flag = (
                is_open
                and is_active
                and day < final_test_start
                and (day_index + item_index * 7) % 47 == 0
            )
            quantity: float | None
            if not is_open:
                quantity = None
                stockout_flag = False
            elif not is_active:
                quantity = 0.0
                stockout_flag = False
            elif stockout_flag:
                quantity = float(np.floor(latent_quantity * 0.55))
            else:
                quantity = float(latent_quantity)
            records.append(
                {
                    "date": day,
                    "item_id": item.item_id,
                    "quantity": quantity,
                    "unit_price": item.base_price,
                    "promo_flag": promo_flag,
                    "stockout_flag": stockout_flag,
                    "store_open": is_open,
                }
            )
    sales = pd.DataFrame.from_records(records)
    for column in ("promo_flag", "stockout_flag", "store_open"):
        sales[column] = sales[column].astype("boolean")
    return sales


def generate_scenario(
    project: ProjectConfig,
    definition: SyntheticScenarioDefinition,
    observed_weather: pd.DataFrame,
) -> DataBundle:
    """Generate a seeded, observed-demand scenario without exposing latent demand."""
    start, end = project.scenario.start_date, project.scenario.end_date
    weather_input = _weather_for_scenario(observed_weather, start, end)
    weather = pd.DataFrame(
        {
            "issued_at": pd.Series(dtype="datetime64[ns]"),
            "target_date": pd.Series(dtype="datetime64[ns]"),
            "temperature": pd.Series(dtype="float64"),
            "rain": pd.Series(dtype="float64"),
            "humidity": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="object"),
        }
    )
    calendar = _calendar_with_tet_closures(start, end)
    daily_sales = _sales_frame(project, definition, calendar, weather_input)
    item_master = pd.DataFrame(
        {
            "item_id": [item.item_id for item in definition.items],
            "item_name": [item.item_name for item in definition.items],
            "category": [item.category for item in definition.items],
            "launch_date": [item.launch_date for item in definition.items],
            "active": [True] * len(definition.items),
        }
    )
    ingredient_master = pd.DataFrame(
        [ingredient.model_dump() for ingredient in definition.ingredients]
    )
    waste_by_ingredient = {
        ingredient.ingredient_id: ingredient.waste_rate for ingredient in definition.ingredients
    }
    recipes = pd.DataFrame(
        [
            {
                **recipe.model_dump(),
                "waste_rate": waste_by_ingredient[recipe.ingredient_id],
            }
            for recipe in definition.recipes
        ]
    )
    received_date = pd.Timestamp(start) - pd.Timedelta(1, unit="D")
    inventory_lots = pd.DataFrame(
        {
            "ingredient_id": [ingredient.ingredient_id for ingredient in definition.ingredients],
            "lot_id": [f"opening-{index + 1:02d}" for index in range(len(definition.ingredients))],
            "on_hand": [ingredient.pack_size * 20 for ingredient in definition.ingredients],
            "expiry_date": [
                received_date + pd.Timedelta(ingredient.shelf_life_days, unit="D")
                for ingredient in definition.ingredients
            ],
            "received_date": [received_date] * len(definition.ingredients),
        }
    )
    scheduled_receipts = pd.DataFrame(
        {
            "ingredient_id": [ingredient.ingredient_id for ingredient in definition.ingredients],
            "arrival_date": [
                pd.Timestamp(start) + pd.Timedelta(ingredient.lead_time_days + 14, unit="D")
                for ingredient in definition.ingredients
            ],
            "quantity": [ingredient.pack_size * 40 for ingredient in definition.ingredients],
        }
    )
    tables = {
        "daily_sales": daily_sales,
        "item_master": item_master,
        "calendar": calendar,
        "weather": weather,
        "recipes": recipes,
        "ingredient_master": ingredient_master,
        "inventory_lots": inventory_lots,
        "scheduled_receipts": scheduled_receipts,
    }
    bundle = validate_bundle(
        DataBundle(
            **tables,
            provenance=DataProvenance(
                name="synthetic_vietnam_scenario",
                source="deterministic_seeded_generator",
                is_synthetic=True,
                generated_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
                checksum=None,
            ),
        )
    )
    provenance = DataProvenance(
        name="synthetic_vietnam_scenario",
        source="deterministic_seeded_generator",
        is_synthetic=True,
        generated_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
        checksum=canonical_bundle_checksum(bundle),
    )
    return DataBundle(
        daily_sales=bundle.daily_sales,
        item_master=bundle.item_master,
        calendar=bundle.calendar,
        weather=bundle.weather,
        recipes=bundle.recipes,
        ingredient_master=bundle.ingredient_master,
        inventory_lots=bundle.inventory_lots,
        scheduled_receipts=bundle.scheduled_receipts,
        provenance=provenance,
    )
