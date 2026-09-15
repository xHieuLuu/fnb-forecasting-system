from pathlib import Path

from typer.testing import CliRunner

from fnb_forecast.cli import _read_selection, _report_root, app


def test_cli_help_exposes_forecast_and_dashboard_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "forecast" in result.output
    assert "simulate-inventory" in result.output


def test_forecast_rejects_unsupported_service_level(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--dataset",
            "synthetic",
            "--origin",
            "2025-12-24",
            "--service-level",
            "0.92",
            "--output",
            str(tmp_path / "forecast"),
        ],
    )

    assert result.exit_code != 0
    assert "service-level" in result.output


def test_evaluate_helpers_resolve_real_store_report_and_selection(tmp_path: Path) -> None:
    report_root = _report_root("real_store_14m")
    assert report_root == Path("artifacts/reports/real_store_calibrated_14m")

    selection = _read_selection("real_store_14m")
    assert selection["selected_model_id"]
    assert selection["selected_run_id"]
