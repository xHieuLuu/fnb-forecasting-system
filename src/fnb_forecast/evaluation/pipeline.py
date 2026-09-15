"""Reusable training, evaluation, and artifact-persistence pipeline."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from fnb_forecast.config import ProjectConfig
from fnb_forecast.contracts import DataBundle, ForecastModel
from fnb_forecast.evaluation.backtest import BacktestRunner
from fnb_forecast.data.io import canonical_bundle_checksum
from fnb_forecast.evaluation.experiment import ExperimentResult, ExperimentRunner
from fnb_forecast.evaluation.uncertainty import ResidualIntervalCalibrator
from fnb_forecast.models import (
    ETSModel,
    LSTMConfig,
    LSTMForecastModel,
    LightGBMConfig,
    LightGBMForecastModel,
    ModelRegistry,
    SeasonalNaiveModel,
    TFTConfig,
    TFTForecastModel,
    config_hash,
)


@dataclass
class PipelineArtifacts:
    result: ExperimentResult
    registry_root: Path
    report_root: Path
    selection_path: Path


def load_yaml_candidates(path: Path, model_type: type[Any]) -> list[Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [model_type.model_validate(candidate) for candidate in payload["candidates"]]


def build_model_factories(config_root: Path, profile: str) -> dict[str, Callable[[], ForecastModel]]:
    """Build the approved candidate set; test profile uses baseline, fast uses one of each family."""
    if profile == "test":
        return {"seasonal_naive_7": lambda: SeasonalNaiveModel()}
    if profile == "fast":
        factories: dict[str, Callable[[], ForecastModel]] = {
            "seasonal_naive_7": lambda: SeasonalNaiveModel(),
            "auto_ets_7": lambda: ETSModel(),
        }
        lgb_candidates = load_yaml_candidates(config_root / "models/lightgbm.yaml", LightGBMConfig)
        for candidate in lgb_candidates[:2]:
            factories[candidate.name] = lambda candidate=candidate: LightGBMForecastModel(candidate)
        lstm_candidates = load_yaml_candidates(config_root / "models/lstm.yaml", LSTMConfig)
        if lstm_candidates:
            candidate = lstm_candidates[0]
            factories[candidate.name] = lambda candidate=candidate: LSTMForecastModel(candidate)
        tft_candidates = load_yaml_candidates(config_root / "models/tft.yaml", TFTConfig)
        if tft_candidates:
            candidate = tft_candidates[0]
            factories[candidate.name] = lambda candidate=candidate: TFTForecastModel(candidate)
        return factories

    factories: dict[str, Callable[[], ForecastModel]] = {
        "seasonal_naive_7": lambda: SeasonalNaiveModel(),
        "auto_ets_7": lambda: ETSModel(),
    }
    for candidate in load_yaml_candidates(config_root / "models/lightgbm.yaml", LightGBMConfig):
        factories[candidate.name] = lambda candidate=candidate: LightGBMForecastModel(candidate)
    for candidate in load_yaml_candidates(config_root / "models/lstm.yaml", LSTMConfig):
        factories[candidate.name] = lambda candidate=candidate: LSTMForecastModel(candidate)
    for candidate in load_yaml_candidates(config_root / "models/tft.yaml", TFTConfig):
        factories[candidate.name] = lambda candidate=candidate: TFTForecastModel(candidate)
    return factories


def run_training_pipeline(
    *,
    dataset_name: str,
    bundle: DataBundle,
    project: ProjectConfig,
    config_path: Path,
    config_root: Path,
    artifact_root: Path,
    profile: str,
) -> PipelineArtifacts:
    """Run validation-first selection, test evaluation, and persist deployable artifacts."""
    factories = build_model_factories(config_root, profile)
    runner = ExperimentRunner(
        backtest_runner=BacktestRunner(include_weather=not bundle.weather.empty),
        model_factories=factories,
    )
    result = runner.run(dataset_name, bundle, project)
    registry_root = artifact_root / "registry"
    report_root = artifact_root / "reports" / dataset_name
    report_root.mkdir(parents=True, exist_ok=True)
    checksum = bundle.provenance.checksum or canonical_bundle_checksum(bundle)
    deploy_history = bundle.daily_sales.copy()
    deploy_builder = _build_deploy_features(bundle)
    deploy_history_features = _deployment_features(bundle, deploy_builder, deploy_history)
    feature_state = deploy_builder.state_dict()
    registry = ModelRegistry(registry_root)

    selected_metadata: dict[str, object] = {}
    for model_id, factory in factories.items():
        model_validation = result.validation_predictions.loc[
            result.validation_predictions["model_id"].eq(model_id)
        ].copy()
        if model_validation.empty:
            continue
        calibrator = ResidualIntervalCalibrator(coverage=0.80).fit(model_validation)
        deploy_model = factory()
        deploy_model.fit(deploy_history_features, deploy_history_features.iloc[0:0].copy())
        is_selected = (model_id == result.selected_model_id)
        run_id = result.selected_run_id if is_selected else f"{model_id}-validation"

        val_wape = 0.0
        if not result.leaderboard.empty and "model_id" in result.leaderboard.columns:
            wape_row = result.leaderboard.loc[result.leaderboard["model_id"].eq(model_id), "validation_wape"]
            if not wape_row.empty:
                val_wape = float(wape_row.iloc[0])

        meta = {
            "dataset_checksum": checksum,
            "provenance": bundle.provenance.name,
            "config_hash": config_hash(project.model_dump()),
            "config_path": str(config_path),
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "train_cutoff": str(pd.to_datetime(deploy_history["date"]).max().date()),
            "features": "full",
            "seeds": [20260813, 20260814, 20260815] if profile == "full" else [20260813],
            "profile": profile,
            "selected": is_selected,
            "validation_wape": val_wape,
        }
        if is_selected:
            selected_metadata = meta

        registry.save(
            dataset_name=dataset_name,
            model_id=model_id,
            run_id=run_id,
            model=deploy_model,
            validation_predictions=model_validation,
            metadata=meta,
            feature_state=feature_state,
            interval_state=calibrator.state_for_json(),
        )
    metadata = selected_metadata
    selection = {
        "selected_model_id": result.selected_model_id,
        "selected_run_id": result.selected_run_id,
        "dataset_name": dataset_name,
        "dataset_checksum": checksum,
        "provenance": bundle.provenance.name,
        "no_weather_model_id": result.selected_model_id,
        "no_weather_run_id": result.selected_run_id,
        "profile": profile,
    }
    selection_path = registry_root / dataset_name / "selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    result.validation_predictions.to_csv(report_root / "validation_predictions.csv", index=False)
    _leaderboard(result.test_predictions).to_csv(report_root / "test_leaderboard.csv", index=False)
    result.test_predictions.to_csv(report_root / "test_predictions.csv", index=False)
    result.leaderboard.to_csv(report_root / "validation_leaderboard.csv", index=False)
    result.leaderboard.to_csv(report_root / "leaderboard.csv", index=False)
    result.ablation_metrics.to_csv(report_root / "ablation_metrics.csv", index=False)
    _fold_metrics(result.test_predictions).to_csv(report_root / "fold_metrics.csv", index=False)
    (report_root / "report.json").write_text(
        json.dumps({**metadata, **selection, "report_root": str(report_root)}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return PipelineArtifacts(result, registry_root, report_root, selection_path)


def _build_deploy_features(bundle: DataBundle) -> Any:
    from fnb_forecast.features import FeatureBuilder

    return FeatureBuilder(calendar=bundle.calendar, weather=bundle.weather, include_weather=False)


def _deployment_features(bundle: DataBundle, builder: Any, history: pd.DataFrame) -> pd.DataFrame:
    from fnb_forecast.features import FeatureBuilder

    origin = pd.to_datetime(history["date"]).max()
    train = FeatureBuilder._training_features(history, origin)
    train = builder._join_calendar(train)
    builder.fit(history)
    return BacktestRunner._add_model_columns(train, bundle)


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["fold_name", "wape", "mae", "bias"])
    from fnb_forecast.evaluation.metrics import forecast_bias, mae, wape

    rows: list[dict[str, object]] = []
    for fold_name, frame in predictions.groupby("fold_name", sort=True):
        actual = frame["actual"].to_numpy(dtype="float64")
        yhat = frame["yhat"].to_numpy(dtype="float64")
        rows.append({"fold_name": fold_name, "wape": wape(actual, yhat), "mae": mae(actual, yhat), "bias": forecast_bias(actual, yhat)})
    return pd.DataFrame(rows)


def _leaderboard(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["model_id", "wape", "mae", "bias"])
    from fnb_forecast.evaluation.metrics import forecast_bias, mae, wape

    rows: list[dict[str, object]] = []
    for model_id, frame in predictions.groupby("model_id", sort=True):
        actual = frame["actual"].to_numpy(dtype="float64")
        yhat = frame["yhat"].to_numpy(dtype="float64")
        rows.append({"model_id": model_id, "wape": wape(actual, yhat), "mae": mae(actual, yhat), "bias": forecast_bias(actual, yhat)})
    return pd.DataFrame(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


