import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from fnb_forecast.config import load_project_config


def test_project_metadata_supports_python_311_and_targets_it() -> None:
    with Path("pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    assert project["project"]["requires-python"] == ">=3.11,<3.13"
    assert project["tool"]["ruff"]["target-version"] == "py311"
    assert project["tool"]["mypy"]["python_version"] == "3.11"


def test_base_config_locks_horizon_and_scenario_dates(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "forecast:\n  horizon_days: 7\n  input_windows: [28, 56]\n"
        "scenario:\n  start_date: 2024-01-02\n  end_date: 2025-12-31\n"
        "  seed: 20260813\n  item_count: 15\n  ingredient_count: 15\n",
        encoding="utf-8",
    )
    config = load_project_config(path)
    assert config.forecast.horizon_days == 7
    assert (config.scenario.end_date - config.scenario.start_date).days + 1 == 730


def test_horizon_other_than_seven_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "forecast:\n  horizon_days: 14\n  input_windows: [28, 56]\n"
        "scenario:\n  start_date: 2024-01-02\n  end_date: 2025-12-31\n"
        "  seed: 1\n  item_count: 15\n  ingredient_count: 15\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_project_config(path)


def test_shifted_730_day_scenario_dates_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "shifted.yaml"
    path.write_text(
        "forecast:\n  horizon_days: 7\n  input_windows: [28, 56]\n"
        "scenario:\n  start_date: 2024-01-03\n  end_date: 2026-01-01\n"
        "  seed: 1\n  item_count: 15\n  ingredient_count: 15\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="scenario dates must be"):
        load_project_config(path)
