"""Integration tests for real store dataset generation and CLI execution."""

from pathlib import Path
from typer.testing import CliRunner
import pandas as pd
import pytest

from fnb_forecast.cli import app

runner = CliRunner()


def test_cli_generate_real_store_and_structure(tmp_path: Path):
    candidates = [
        Path("dataset.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset_14_thang_thuc_te.xlsx"),
        Path("f:/Projects/ForecastingSystem/dataset_real_store_14m_monthly.xlsx"),
        Path("dataset_14_thang_thuc_te.xlsx"),
    ]
    excel_path = next((p for p in candidates if p.exists()), candidates[0])
        
    output_dir = tmp_path / "real_store_14m"
    result = runner.invoke(app, [
        "generate-real-store",
        "--excel", str(excel_path),
        "--output", str(output_dir),
    ])
    
    assert result.exit_code == 0, result.output
    assert "Generated 426 days across 43 items" in result.output
    
    # Verify all 8 files and provenance
    expected_files = [
        "daily_sales.csv", "item_master.csv", "calendar.csv", "weather.csv",
        "recipes.csv", "ingredient_master.csv", "inventory_lots.csv",
        "scheduled_receipts.csv", "provenance.json"
    ]
    for fname in expected_files:
        assert (output_dir / fname).exists(), f"Missing {fname}"
        
    df_sales = pd.read_csv(output_dir / "daily_sales.csv")
    assert len(df_sales) == 43 * 426
