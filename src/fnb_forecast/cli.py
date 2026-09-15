"""Typer commands for the offline F&B forecasting workflow."""

from __future__ import annotations
import os
import warnings

# Suppress repetitive UserWarnings (e.g. StandardScaler feature names, PyTorch CuBLAS deterministic)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import shutil
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from fnb_forecast.config import load_project_config
from fnb_forecast.contracts import DataBundle, DataProvenance, ForecastRequest
from fnb_forecast.data.io import canonical_bundle_checksum, load_csv_bundle, save_bundle
from fnb_forecast.data.m5 import load_m5
from fnb_forecast.data.synthetic import generate_scenario, load_synthetic_definition
from fnb_forecast.evaluation.uncertainty import ResidualIntervalCalibrator
from fnb_forecast.planning import InventorySimulator, calculate_ingredient_need
from fnb_forecast.data.validation import validate_bundle
from fnb_forecast.evaluation.pipeline import run_training_pipeline
from fnb_forecast.data.weather import OpenMeteoWeatherClient
from fnb_forecast.features import FeatureBuilder
from fnb_forecast.models.registry import ModelRegistry
from fnb_forecast.services import ForecastService

app = typer.Typer(help="F&B time-series forecasting commands.")
SUPPORTED_DATASETS = {"synthetic", "m5", "real_store_14m"}
SERVICE_LEVELS = {0.90, 0.95, 0.98}
DATASET_NAMES = {
    "synthetic": "synthetic_vietnam_scenario",
    "m5": "m5_public_benchmark",
    "real_store_14m": "real_store_calibrated_14m",
}


@app.callback()
def main() -> None:
    """F&B time-series forecasting commands."""


@app.command("generate-real-store")
def generate_real_store(
    excel: Path = typer.Option(..., exists=True, readable=True, help="Path to dataset.xlsx"),
    output: Path = typer.Option(..., help="Target directory for processed 8-table bundle"),
    seed: int = typer.Option(42, help="Random seed for calibration"),
) -> None:
    """Generate and save the 14-month calibrated real store DataBundle."""
    from fnb_forecast.data.real_store_generator import generate_real_store_14m_bundle, save_real_store_bundle
    bundle = generate_real_store_14m_bundle(excel, seed=seed)
    save_real_store_bundle(bundle, output)
    days = len(bundle.calendar)
    items = len(bundle.item_master)
    typer.echo(f"Generated {days} days across {items} items at {output}")


@app.command("generate-synthetic")
def generate_synthetic(
    project_config: Path = typer.Option(..., exists=True, readable=True),
    scenario_config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
) -> None:
    """Generate and save the deterministic Vietnam F&B scenario."""
    project = load_project_config(project_config)
    definition = load_synthetic_definition(scenario_config)
    weather_client = OpenMeteoWeatherClient(
        project.location.latitude,
        project.location.longitude,
        cache_root=project.paths.cache_root,
    )
    observed_weather = weather_client.fetch_historical_observations(
        project.scenario.start_date, project.scenario.end_date
    )
    bundle = generate_scenario(project, definition, observed_weather)
    save_bundle(validate_bundle(bundle), output)
    days = len(bundle.daily_sales) // project.scenario.item_count
    typer.echo(f"Generated {days} days at {output}")


@app.command("prepare-m5")
def prepare_m5(
    raw_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    selection_end: str = typer.Option(...),
    output: Path = typer.Option(...),
    store_id: str = typer.Option("CA_1"),
    n_items: int = typer.Option(20, min=1),
) -> None:
    """Save a leakage-safe public M5 subset without downloading data."""
    bundle = load_m5(raw_dir, store_id, n_items, pd.Timestamp(selection_end))
    save_bundle(bundle, output)
    typer.echo(f"Prepared {n_items} M5 items at {output}")


@app.command("train")
def train(
    dataset: str = typer.Option(...),
    config: Path = typer.Option(..., exists=True, readable=True),
    data_dir: Path | None = typer.Option(None, help="Override data bundle path."),
    profile: str = typer.Option("full"),
) -> None:
    """Run validation-first training and persist registry/report artifacts."""
    typer.echo(f"[*] Training models for dataset: '{dataset}' (Profile: {profile})...")
    project = load_project_config(config)
    if dataset not in SUPPORTED_DATASETS:
        raise typer.BadParameter("dataset must be synthetic, m5, or real_store_14m")
    if profile not in {"full", "test", "fast"}:
        raise typer.BadParameter("profile must be full, fast, or test")
    bundle_root = _discover_bundle(dataset, data_dir)
    typer.echo(f"[*] Loading bundle from: {bundle_root}...")
    bundle = _load_bundle_with_provenance(bundle_root, dataset)
    typer.echo(f"[*] Running backtest and model evaluation...")
    artifacts = run_training_pipeline(
        dataset_name=DATASET_NAMES[dataset], bundle=bundle, project=project,
        config_path=config, config_root=Path(__file__).parents[2] / "configs",
        artifact_root=project.paths.artifact_root, profile=profile,
    )
    typer.echo(f"[+] Training complete; selection written to {artifacts.selection_path}")



def _report_root(dataset: str) -> Path:
    """Resolve the persisted report directory for a supported dataset."""
    if dataset not in SUPPORTED_DATASETS:
        raise typer.BadParameter(f"unsupported dataset: {dataset}")
    return Path("artifacts") / "reports" / DATASET_NAMES[dataset]


def _read_selection(dataset: str) -> dict[str, str]:
    """Read the frozen model selection metadata for a supported dataset."""
    if dataset not in SUPPORTED_DATASETS:
        return {}
    path = Path("artifacts") / "registry" / DATASET_NAMES[dataset] / "selection.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        key: str(payload[key])
        for key in ("selected_model_id", "selected_run_id")
        if payload.get(key) is not None
    }


@app.command("evaluate")
def evaluate(
    dataset: str = typer.Option(...),
    role: str = typer.Option(...),
    selected_only: bool = typer.Option(False),
) -> None:
    """Read persisted validation/test metrics without mixing selector inputs."""
    if role not in {"validation", "test"}:
        raise typer.BadParameter("role must be validation or test")

    root = _report_root(dataset)
    name = "validation_leaderboard.csv" if role == "validation" else "test_leaderboard.csv"
    path = root / name
    if not path.exists():
        raise typer.BadParameter(f"report not found: {path}; run train first")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        frame = pd.DataFrame()
    selection = _read_selection(dataset)
    if selected_only and "model_id" in frame.columns and selection:
        frame = frame.loc[frame["model_id"].eq(selection.get("selected_model_id"))]
    typer.echo(json.dumps({"dataset": dataset, "role": role, "selected_only": selected_only, "rows": frame.to_dict(orient="records")}, sort_keys=True, default=str))

@app.command("report")
def report(dataset: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Write a compact machine-readable report manifest."""
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(
            {"dataset": dataset, "validation_and_test_separate": True, "report": str(report_path)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Report written to {report_path}")


@app.command("simulate-inventory")
def simulate_inventory(
    dataset: str = typer.Option(...),
    service_level: float = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Write inventory simulation output placeholders for a selected artifact dataset."""
    if service_level not in SERVICE_LEVELS:
        raise typer.BadParameter("service-level must be one of 0.90, 0.95, or 0.98")


@app.command("forecast")
def forecast(
    dataset: str = typer.Option(...),
    origin: str = typer.Option(...),
    service_level: float = typer.Option(...),
    output: Path = typer.Option(...),
    data_dir: Path | None = typer.Option(
        None, help="Override data/processed/<dataset> bundle path."
    ),
    registry: Path | None = typer.Option(None, help="Override model registry root."),
) -> None:
    """Run a seven-day forecast and export auditable decision tables."""
    if dataset not in SUPPORTED_DATASETS:
        raise typer.BadParameter("dataset must be synthetic or m5")
    if service_level not in SERVICE_LEVELS:
        raise typer.BadParameter("service-level must be one of 0.90, 0.95, or 0.98")
    try:
        forecast_origin = pd.Timestamp(origin).normalize()
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(f"origin must be an ISO date: {error}") from error

    bundle_root = data_dir or (Path("data") / "processed" / dataset)
    bundle = _load_bundle_with_provenance(bundle_root, dataset)
    registry_root = registry or (Path("artifacts") / "registry")
    selected = _resolve_selection(dataset, registry_root)
    feature_builder = FeatureBuilder(calendar=bundle.calendar, weather=bundle.weather)
    no_weather_builder = (
        FeatureBuilder(calendar=bundle.calendar, weather=bundle.weather, include_weather=False)
        if selected.get("no_weather_model_id")
        else None
    )
    service = ForecastService(
        registry=ModelRegistry(registry_root),
        feature_builder=feature_builder,
        selected_model_id=selected["selected_model_id"],
        selected_run_id=selected["selected_run_id"],
        interval_calibrator=_load_interval_calibrator(registry_root, DATASET_NAMES[dataset], selected),
        no_weather_feature_builder=no_weather_builder,
        no_weather_model_id=selected.get("no_weather_model_id"),
        no_weather_run_id=selected.get("no_weather_run_id"),
    )
    request = ForecastRequest(
        forecast_origin=forecast_origin,
        horizon_days=7,
        dataset_name=DATASET_NAMES[dataset],
    )
    result = service.run(request, bundle, service_level=service_level)
    output.mkdir(parents=True, exist_ok=True)
    result.item_forecasts.to_csv(output / "item_forecasts.csv", index=False)
    result.revenue_forecast.to_csv(output / "revenue_forecast.csv", index=False)
    result.ingredient_forecast.to_csv(output / "ingredient_forecast.csv", index=False)
    result.order_proposals.to_csv(output / "order_proposals.csv", index=False)
    (output / "run_metadata.json").write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    typer.echo(f"Forecast exports written to {output}")


def _discover_bundle(dataset: str, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    candidates = [
        Path("data") / "processed" / dataset,
        Path("data") / "processed" / DATASET_NAMES[dataset],
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise typer.BadParameter(f"data bundle not found; tried: {candidates}")


def _load_bundle_with_provenance(root: Path, dataset: str) -> DataBundle:
    root = Path(root)
    provenance_path = root / "provenance.json"
    if not root.is_dir():
        raise typer.BadParameter(f"data bundle not found: {root}")
    if not provenance_path.exists():
        raise typer.BadParameter(f"provenance.json not found: {provenance_path}")
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        generated_at = payload.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        provenance = DataProvenance(
            name=str(payload.get("name", DATASET_NAMES[dataset])),
            source=str(payload.get("source", "local_csv_bundle")),
            is_synthetic=bool(payload.get("is_synthetic", dataset == "synthetic")),
            generated_at=generated_at,
            checksum=payload.get("checksum"),
        )
        bundle = load_csv_bundle(root, provenance)
        if bundle.provenance.checksum is None:
            bundle.provenance = DataProvenance(
                name=bundle.provenance.name,
                source=bundle.provenance.source,
                is_synthetic=bundle.provenance.is_synthetic,
                generated_at=bundle.provenance.generated_at,
                checksum=canonical_bundle_checksum(bundle),
            )
        return bundle
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"invalid data bundle at {root}: {error}") from error


def _resolve_selection(dataset: str, registry_root: Path) -> dict[str, str]:
    """Read frozen selection metadata, falling back to the local Seasonal Naive baseline."""
    for path in (
        Path(registry_root) / DATASET_NAMES[dataset] / "selection.json",
        Path("artifacts") / "runs" / dataset / "selection.json",
        Path("artifacts") / "runs" / DATASET_NAMES[dataset] / "selection.json",
        Path("artifacts") / "runs" / dataset / "selected_model.json",
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            selected: dict[str, str] = {}
            for key in (
                "selected_model_id",
                "selected_run_id",
                "no_weather_model_id",
                "no_weather_run_id",
            ):
                if payload.get(key) is not None:
                    selected[key] = str(payload[key])
            if "selected_model_id" in selected:
                return selected
        except (OSError, json.JSONDecodeError):
            continue

    selected = {
        "selected_model_id": "seasonal_naive_7",
        "selected_run_id": "seasonal_naive_7-validation",
    }
    dataset_root = Path(registry_root) / DATASET_NAMES[dataset]
    metadata_candidates = sorted(dataset_root.glob("*/*/metadata.json"))
    for metadata_path in metadata_candidates:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("selected") is True:
            selected["selected_model_id"] = str(payload.get("model_id"))
            selected["selected_run_id"] = str(payload.get("run_id"))
            break
    return selected
def _load_interval_calibrator(
    registry_root: Path, dataset_name: str, selection: dict[str, str]
) -> ResidualIntervalCalibrator | None:
    model_id = selection.get("selected_model_id")
    run_id = selection.get("selected_run_id")
    if not model_id or not run_id:
        return None
    path = Path(registry_root) / dataset_name / model_id / run_id / "interval_state.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibrator = ResidualIntervalCalibrator.from_state(payload)
        if calibrator.state.get("quantiles"):
            return calibrator

        run_root = path.parent
        validation_path = run_root / "validation_predictions.parquet"
        if validation_path.exists():
            validation_predictions = pd.read_parquet(validation_path)
        else:
            validation_path = run_root / "validation_predictions.csv"
            if not validation_path.exists():
                return None
            validation_predictions = pd.read_csv(validation_path)
        return ResidualIntervalCalibrator(coverage=0.80).fit(validation_predictions)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None

if __name__ == "__main__":
    app()
