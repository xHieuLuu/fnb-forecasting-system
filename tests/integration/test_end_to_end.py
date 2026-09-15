import json
from pathlib import Path

from typer.testing import CliRunner

from fnb_forecast.cli import app
from fnb_forecast.data.weather import OpenMeteoWeatherClient


def test_synthetic_csv_to_decision_exports(
    tmp_path: Path,
    monkeypatch,
    project_config,
    scenario_definition,
    observed_weather,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        OpenMeteoWeatherClient,
        "fetch_historical_observations",
        lambda _self, _start, _end: observed_weather,
    )
    root = Path(__file__).parents[2]
    bundle_dir = tmp_path / "bundle"
    runner = CliRunner()
    generated = runner.invoke(
        app,
        [
            "generate-synthetic",
            "--project-config",
            str(root / "configs/base.yaml"),
            "--scenario-config",
            str(root / "configs/synthetic.yaml"),
            "--output",
            str(bundle_dir),
        ],
    )
    assert generated.exit_code == 0, generated.output

    trained = runner.invoke(
        app,
        [
            "train",
            "--dataset",
            "synthetic",
            "--config",
            str(root / "configs/base.yaml"),
            "--profile",
            "test",
            "--data-dir",
            str(bundle_dir),
        ],
    )
    assert trained.exit_code == 0, trained.output

    output_dir = tmp_path / "forecasts" / "2025-12-24"
    forecasted = runner.invoke(
        app,
        [
            "forecast",
            "--dataset",
            "synthetic",
            "--origin",
            "2025-12-24",
            "--service-level",
            "0.95",
            "--data-dir",
            str(bundle_dir),
            "--registry",
            str(tmp_path / "registry"),
            "--output",
            str(output_dir),
        ],
    )
    assert forecasted.exit_code == 0, forecasted.output
    for name in (
        "item_forecasts.csv",
        "revenue_forecast.csv",
        "ingredient_forecast.csv",
        "order_proposals.csv",
        "run_metadata.json",
    ):
        assert (output_dir / name).exists()
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["provenance"] == "synthetic_vietnam_scenario"
    assert metadata["horizon_days"] == 7
