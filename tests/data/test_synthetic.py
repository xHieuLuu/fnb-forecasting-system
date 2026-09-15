from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from fnb_forecast.cli import app
from fnb_forecast.data.io import canonical_bundle_checksum, load_csv_bundle, save_bundle
from fnb_forecast.data.synthetic import generate_scenario
from fnb_forecast.data.validation import validate_bundle
from fnb_forecast.data.weather import assert_weather_known_at_origin


def test_scenario_has_exact_scope_and_is_deterministic(
    project_config, scenario_definition, observed_weather
) -> None:
    first = generate_scenario(project_config, scenario_definition, observed_weather)
    second = generate_scenario(project_config, scenario_definition, observed_weather)

    assert first.item_master["item_id"].nunique() == 15
    assert first.ingredient_master["ingredient_id"].nunique() == 15
    assert len(first.daily_sales) == 15 * 730
    assert first.daily_sales.equals(second.daily_sales)
    assert first.provenance.name == "synthetic_vietnam_scenario"
    assert first.provenance.is_synthetic is True


def test_synthetic_sales_are_nonnegative_and_closed_days_are_nan(
    project_config, scenario_definition, observed_weather
) -> None:
    bundle = generate_scenario(project_config, scenario_definition, observed_weather)

    open_rows = bundle.daily_sales["store_open"]
    assert bundle.daily_sales.loc[open_rows, "quantity"].ge(0).all()
    assert bundle.daily_sales.loc[~open_rows, "quantity"].isna().all()
    final_test_start = bundle.daily_sales["date"].max() - pd.Timedelta(27, unit="D")
    assert not bundle.daily_sales.loc[
        bundle.daily_sales["date"].ge(final_test_start), "stockout_flag"
    ].any()
    assert "latent_quantity" not in bundle.daily_sales.columns


def test_historical_observations_are_not_exposed_as_future_weather_covariates(
    project_config, scenario_definition, observed_weather
) -> None:
    bundle = generate_scenario(project_config, scenario_definition, observed_weather)
    origin = bundle.daily_sales["date"].min() - pd.Timedelta(1, unit="D")

    assert bundle.weather.empty
    assert_weather_known_at_origin(bundle.weather, origin)
    assert not bundle.weather["target_date"].isin(observed_weather["date"]).any()


def test_generator_creates_tet_closures_launches_and_inventory_metadata(
    project_config, scenario_definition, observed_weather
) -> None:
    bundle = generate_scenario(project_config, scenario_definition, observed_weather)

    tet_closures = bundle.calendar.loc[
        bundle.calendar["days_to_tet"].isin([0, 1]) & ~bundle.calendar["store_open"]
    ]
    assert tet_closures.groupby(tet_closures["date"].dt.year).size().to_dict() == {2024: 2, 2025: 2}
    assert bundle.item_master["launch_date"].nunique() == 3
    assert bundle.inventory_lots["expiry_date"].notna().all()
    assert not bundle.scheduled_receipts.empty
    assert set(bundle.recipes["ingredient_id"]) == set(bundle.ingredient_master["ingredient_id"])


def test_generate_synthetic_command_saves_validated_bundle(
    tmp_path, monkeypatch, observed_weather
) -> None:
    from fnb_forecast.data.weather import OpenMeteoWeatherClient

    monkeypatch.setattr(
        OpenMeteoWeatherClient,
        "fetch_historical_observations",
        lambda _self, _start, _end: observed_weather,
    )
    result = CliRunner().invoke(
        app,
        [
            "generate-synthetic",
            "--project-config",
            str(Path("configs/base.yaml")),
            "--scenario-config",
            str(Path("configs/synthetic.yaml")),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    for table in (
        "daily_sales",
        "item_master",
        "calendar",
        "weather",
        "recipes",
        "ingredient_master",
        "inventory_lots",
        "scheduled_receipts",
    ):
        assert (tmp_path / f"{table}.csv").exists()
    assert (tmp_path / "provenance.json").exists()


def test_saved_validated_tables_match_the_scenario_provenance_checksum(
    tmp_path, project_config, scenario_definition, observed_weather
) -> None:
    generated = generate_scenario(project_config, scenario_definition, observed_weather)
    validated = validate_bundle(generated)
    save_bundle(validated, tmp_path)
    loaded = load_csv_bundle(tmp_path, validated.provenance)

    assert canonical_bundle_checksum(loaded) == validated.provenance.checksum


def test_scenario_definition_rejects_unknown_yaml_fields(tmp_path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("items: []\ningredients: []\nrecipes: []\nunknown: true\n", encoding="utf-8")

    from fnb_forecast.data.synthetic import load_synthetic_definition

    with pytest.raises(ValidationError):
        load_synthetic_definition(config)
