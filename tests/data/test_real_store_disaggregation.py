"""Unit tests for the real store constrained integer disaggregation engine."""

import pandas as pd
import numpy as np
import pytest

from fnb_forecast.data.real_store_disaggregation import (
    disaggregate_daily_revenue,
    disaggregate_revenue_series,
    get_category_priors,
)
from fnb_forecast.data.real_store_master import get_real_store_item_master


def test_disaggregate_daily_revenue_exact_match():
    items = get_real_store_item_master()
    prices = {row["item_id"]: float(row["unit_price"]) for _, row in items.iterrows()}
    priors = get_category_priors(items, is_weekend=False)
    
    test_targets = [50000.0, 70000.0, 180000.0, 267000.0, 300000.0, 585000.0]
    for target in test_targets:
        quantities = disaggregate_daily_revenue(target, prices, priors, seed=42)
        total_calc = sum(quantities.get(item_id, 0) * prices[item_id] for item_id in prices)
        assert total_calc == target, f"Expected {target}, got {total_calc}"
        assert all(q >= 0 and isinstance(q, int) for q in quantities.values())


def test_disaggregate_revenue_series_and_totals():
    df_rev = pd.DataFrame({
        "date": pd.date_range("2025-05-01", periods=5),
        "revenue": [300000.0, 330000.0, 450000.0, 300000.0, 267000.0]
    })
    
    sales_df = disaggregate_revenue_series(df_rev, seed=123)
    assert set(sales_df.columns) >= {"date", "item_id", "quantity", "unit_price", "store_open", "promo_flag", "stockout_flag"}
    
    # Check total for each date matches exact input revenue
    for date, group in sales_df.groupby("date"):
        calc_rev = (group["quantity"] * group["unit_price"]).sum()
        expected_rev = df_rev[df_rev["date"] == date]["revenue"].iloc[0]
        assert calc_rev == expected_rev
