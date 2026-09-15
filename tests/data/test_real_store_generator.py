"""Unit tests for the 14-month real store DataBundle generator."""

from pathlib import Path
import pandas as pd
import pytest

from fnb_forecast.data.real_store_generator import (
    generate_real_store_14m_bundle,
    save_real_store_bundle,
)
from fnb_forecast.data.validation import validate_bundle
from fnb_forecast.data.io import load_csv_bundle


def _find_excel_path() -> Path:
    candidates = [
        Path("dataset.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset_14_thang_thuc_te.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset_real_store_14m_monthly.xlsx"),
        Path("dataset_14_thang_thuc_te.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def test_generate_real_store_14m_bundle_validates():
    excel_path = _find_excel_path()
        
    bundle = generate_real_store_14m_bundle(excel_path, seed=42)
    
    # 1. Check calendar: exactly 426 days (2025-05-01 to 2026-06-30)
    assert len(bundle.calendar) == 426
    assert bundle.calendar["date"].min() == pd.Timestamp("2025-05-01")
    assert bundle.calendar["date"].max() == pd.Timestamp("2026-06-30")
    
    # 2. Check daily sales shape: 43 items * 426 days = 18,318 rows
    assert len(bundle.daily_sales) == 43 * 426
    
    # 3. Check Anchor months match Excel
    may_sales = bundle.daily_sales[bundle.daily_sales["date"].dt.month == 5]
    may_sales_2025 = may_sales[may_sales["date"].dt.year == 2025]
    may_rev = (may_sales_2025["quantity"].fillna(0) * may_sales_2025["unit_price"]).sum()
    assert may_rev == 7572000
    
    june_sales = bundle.daily_sales[bundle.daily_sales["date"].dt.month == 6]
    june_sales_2025 = june_sales[june_sales["date"].dt.year == 2025]
    june_rev = (june_sales_2025["quantity"].fillna(0) * june_sales_2025["unit_price"]).sum()
    assert june_rev == 7858000
    
    # 4. Strict Schema & Contract Validation
    validated = validate_bundle(bundle)
    assert validated is not None


def test_save_and_load_real_store_bundle(tmp_path: Path):
    excel_path = _find_excel_path()
        
    bundle = generate_real_store_14m_bundle(excel_path, seed=42)
    out_dir = tmp_path / "real_store_14m"
    save_real_store_bundle(bundle, out_dir)
    
    assert (out_dir / "daily_sales.csv").exists()
    assert (out_dir / "item_master.csv").exists()
    assert (out_dir / "recipes.csv").exists()
    assert (out_dir / "ingredient_master.csv").exists()
    assert (out_dir / "calendar.csv").exists()
    assert (out_dir / "weather.csv").exists()
    assert (out_dir / "inventory_lots.csv").exists()
    assert (out_dir / "scheduled_receipts.csv").exists()
    assert (out_dir / "provenance.json").exists()
    
    loaded = load_csv_bundle(out_dir, bundle.provenance)
    assert len(loaded.daily_sales) == 43 * 426
