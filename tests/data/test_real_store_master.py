"""Unit tests for real store master data tables and Excel parsing."""

from pathlib import Path
import pandas as pd
import pytest

from fnb_forecast.data.real_store_master import (
    get_real_store_item_master,
    get_real_store_ingredient_master,
    get_real_store_recipes,
    parse_real_store_excel,
)


def test_item_master_schema_and_counts():
    items = get_real_store_item_master()
    assert len(items) == 43
    assert set(items.columns) == {"item_id", "item_name", "category", "launch_date", "active", "unit_price"}
    assert items["item_id"].is_unique
    assert (items["unit_price"] > 0).all()
    assert items["active"].all()


def test_ingredient_master_and_recipes_consistency():
    ingredients = get_real_store_ingredient_master()
    recipes = get_real_store_recipes()
    
    assert set(ingredients.columns) == {
        "ingredient_id", "name", "base_unit", "pack_size", "unit_cost",
        "lead_time_days", "review_period_days", "shelf_life_days"
    }
    assert ingredients["ingredient_id"].is_unique
    assert (ingredients["unit_cost"] > 0).all()
    
    assert set(recipes.columns) == {"item_id", "ingredient_id", "amount", "unit", "waste_rate"}
    assert set(recipes["ingredient_id"]).issubset(set(ingredients["ingredient_id"]))
    assert set(recipes["item_id"]).issubset(set(get_real_store_item_master()["item_id"]))
    assert (recipes["amount"] > 0).all()
    assert (recipes["waste_rate"] >= 0).all()


def test_parse_real_store_excel():
    candidates = [
        Path("dataset.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset_14_thang_thuc_te.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset_real_store_14m_monthly.xlsx"),
        Path("dataset_14_thang_thuc_te.xlsx"),
    ]
    excel_path = next((p for p in candidates if p.exists()), candidates[0])
    
    daily_rev, daily_ice = parse_real_store_excel(excel_path)
    assert not daily_rev.empty
    assert "date" in daily_rev.columns and "revenue" in daily_rev.columns
    
    # May 2025: 29 days with positive revenue, sum = 7,572,000
    may_rev = daily_rev[daily_rev["date"].dt.month == 5]["revenue"].sum()
    assert may_rev == 7572000
    
    # June 2025: 28 days with positive revenue, sum = 7,858,000
    june_rev = daily_rev[daily_rev["date"].dt.month == 6]["revenue"].sum()
    assert june_rev == 7858000
    
    assert not daily_ice.empty
    assert "date" in daily_ice.columns and "ice_cost" in daily_ice.columns
