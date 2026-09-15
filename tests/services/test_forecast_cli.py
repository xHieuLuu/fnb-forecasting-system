from dataclasses import replace
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fnb_forecast.cli import app
from fnb_forecast.data.io import save_bundle


def test_forecast_command_writes_decision_exports(tmp_path: Path, valid_bundle) -> None:
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    calendar = pd.DataFrame(
        {
            "date": dates,
            "weekday": dates.weekday,
            "is_weekend": pd.Series(dates.weekday >= 5, dtype="boolean"),
            "holiday_name": [None] * len(dates),
            "days_to_tet": range(30, 20, -1),
            "days_after_tet": [None] * len(dates),
            "store_open": pd.Series(True, index=range(len(dates)), dtype="boolean"),
        }
    )
    bundle_dir = tmp_path / "bundle"
    save_bundle(replace(valid_bundle, calendar=calendar), bundle_dir)
    output = tmp_path / "forecast"

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--dataset",
            "synthetic",
            "--origin",
            "2025-01-03",
            "--service-level",
            "0.95",
            "--data-dir",
            str(bundle_dir),
            "--registry",
            str(tmp_path / "registry"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    for name in (
        "item_forecasts.csv",
        "revenue_forecast.csv",
        "ingredient_forecast.csv",
        "order_proposals.csv",
        "run_metadata.json",
    ):
        assert (output / name).exists()