"""Shared dashboard state and offline artifact loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from fnb_forecast.cli import (
    DATASET_NAMES,
    _load_bundle_with_provenance,
    _load_interval_calibrator,
    _resolve_selection,
)
from fnb_forecast.contracts import DataBundle, ForecastRequest
from fnb_forecast.features import FeatureBuilder
from fnb_forecast.models.registry import ModelRegistry
from fnb_forecast.planning.revenue import PlanningOutput
from fnb_forecast.services import ForecastService

DATASET_OPTIONS: dict[str, str] = {
    "real_store_14m": "Quán thực tế (14 tháng - 2025/2026)",
}
DEFAULT_DATASET = "real_store_14m"

_EMPTY_ITEM_COLUMNS = [
    "target_date",
    "item_id",
    "yhat",
    "lower",
    "upper",
    "model_id",
    "fallback_used",
    "fallback_reason",
]
_EMPTY_REVENUE_COLUMNS = ["target_date", "item_id", "yhat", "unit_price", "forecast_revenue"]
_EMPTY_INGREDIENT_COLUMNS = ["target_date", "ingredient_id", "unit", "forecast_need"]
_EMPTY_ORDER_COLUMNS = ["ingredient_id", "suggested_order", "warning"]


def empty_planning_output() -> PlanningOutput:
    """Return a renderable empty result for a fresh dashboard session."""
    return PlanningOutput(
        item_forecasts=pd.DataFrame(columns=_EMPTY_ITEM_COLUMNS),
        revenue_forecast=pd.DataFrame(columns=_EMPTY_REVENUE_COLUMNS),
        ingredient_forecast=pd.DataFrame(columns=_EMPTY_INGREDIENT_COLUMNS),
        ingredient_detail=pd.DataFrame(),
        order_proposals=pd.DataFrame(columns=_EMPTY_ORDER_COLUMNS),
        warnings=["Chưa có dữ liệu dự báo; hãy chọn ngày và bấm Chạy dự báo."],
        metadata={"provenance": "real_store_14m", "horizon_days": 7},
    )


def enrich_output_with_bundle(output: PlanningOutput, bundle: DataBundle | None) -> PlanningOutput:
    """Add friendly item names, categories, and ingredient names from bundle to output."""
    if bundle is None:
        return output

    # Enrich item forecasts and revenue forecasts
    if not bundle.item_master.empty:
        cols = [c for c in ["item_id", "item_name", "category"] if c in bundle.item_master.columns]
        item_meta = bundle.item_master.loc[:, cols].drop_duplicates()

        if not output.item_forecasts.empty:
            missing_cols = [c for c in ["item_name", "category"] if c in item_meta.columns and c not in output.item_forecasts.columns]
            if missing_cols:
                output.item_forecasts = output.item_forecasts.merge(item_meta[["item_id"] + missing_cols], on="item_id", how="left")
            if "item_name" in output.item_forecasts.columns:
                output.item_forecasts["item_name"] = output.item_forecasts["item_name"].fillna(output.item_forecasts["item_id"])
            if "category" in output.item_forecasts.columns:
                output.item_forecasts["category"] = output.item_forecasts["category"].fillna("Khác")

        if not output.revenue_forecast.empty:
            missing_cols = [c for c in ["item_name", "category"] if c in item_meta.columns and c not in output.revenue_forecast.columns]
            if missing_cols:
                output.revenue_forecast = output.revenue_forecast.merge(item_meta[["item_id"] + missing_cols], on="item_id", how="left")
            if "item_name" in output.revenue_forecast.columns:
                output.revenue_forecast["item_name"] = output.revenue_forecast["item_name"].fillna(output.revenue_forecast["item_id"])

    # Enrich ingredient forecasts and order proposals
    if not bundle.ingredient_master.empty:
        ing_meta = bundle.ingredient_master.copy()
        if "name" in ing_meta.columns and "ingredient_name" not in ing_meta.columns:
            ing_meta = ing_meta.rename(columns={"name": "ingredient_name"})
        cols_to_use = [c for c in ["ingredient_id", "ingredient_name", "base_unit", "pack_size", "shelf_life_days"] if c in ing_meta.columns]
        ing_meta = ing_meta.loc[:, cols_to_use].drop_duplicates()

        if not output.ingredient_forecast.empty and "ingredient_name" not in output.ingredient_forecast.columns:
            output.ingredient_forecast = output.ingredient_forecast.merge(ing_meta, on="ingredient_id", how="left")
            output.ingredient_forecast["ingredient_name"] = output.ingredient_forecast["ingredient_name"].fillna(output.ingredient_forecast["ingredient_id"])

        if not output.order_proposals.empty and "ingredient_name" not in output.order_proposals.columns:
            output.order_proposals = output.order_proposals.merge(ing_meta, on="ingredient_id", how="left")
            output.order_proposals["ingredient_name"] = output.order_proposals["ingredient_name"].fillna(output.order_proposals["ingredient_id"])

    return output


def get_latest_output() -> PlanningOutput:
    """Load the most recent CSV decision output, or return a safe empty state."""
    cached = st.session_state.get("dashboard_output")
    if isinstance(cached, PlanningOutput):
        return cached
    root = Path(os.environ.get("FNB_FORECAST_OUTPUT_ROOT", "artifacts/forecasts"))
    candidates = sorted(root.glob("*/run_metadata.json")) if root.exists() else []
    if not candidates:
        return empty_planning_output()
    metadata_path = candidates[-1]
    output_root = metadata_path.parent
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prov = metadata.get("provenance", metadata.get("dataset_name", DEFAULT_DATASET))
        dataset_key = DEFAULT_DATASET
        for key in DATASET_OPTIONS:
            if key in str(prov) or DATASET_NAMES.get(key) in str(prov):
                dataset_key = key
                break
        
        output = PlanningOutput(
            item_forecasts=_read_csv(output_root / "item_forecasts.csv", _EMPTY_ITEM_COLUMNS),
            revenue_forecast=_read_csv(
                output_root / "revenue_forecast.csv", _EMPTY_REVENUE_COLUMNS
            ),
            ingredient_forecast=_read_csv(
                output_root / "ingredient_forecast.csv", _EMPTY_INGREDIENT_COLUMNS
            ),
            ingredient_detail=pd.DataFrame(),
            order_proposals=_read_csv(output_root / "order_proposals.csv", _EMPTY_ORDER_COLUMNS),
            warnings=[str(value) for value in metadata.get("warnings", [])],
            metadata=metadata,
        )
        bundle = get_bundle(dataset_key)
        return enrich_output_with_bundle(output, bundle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        result = empty_planning_output()
        result.warnings = [f"Không đọc được forecast output: {error}"]
        return result


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)


@st.cache_resource(show_spinner=False)
def load_bundle_cached(root: str, dataset: str) -> DataBundle:
    """Cache validated immutable-on-load bundle data for dashboard reruns."""
    return _load_bundle_with_provenance(Path(root), dataset)


def get_bundle(dataset: str | None = None) -> DataBundle | None:
    if dataset is None:
        dataset = st.session_state.get("selected_dataset", DEFAULT_DATASET)
    root = Path(os.environ.get("FNB_FORECAST_BUNDLE", "data/processed")) / dataset
    if not root.exists():
        root = Path("data/processed") / dataset
        if not root.exists():
            return None
    try:
        return load_bundle_cached(str(root), dataset)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def get_model_options(dataset: str = DEFAULT_DATASET) -> dict[str, str]:
    """Return map of model_id -> friendly label for dashboard sidebar UI."""
    from fnb_forecast.app.components import friendly_model_name

    options = {
        "auto": "Tự động chọn",
        "lgb_l15_d-1_lr0.03": friendly_model_name("lgb_l15_d-1_lr0.03"),
        "seasonal_naive_7": friendly_model_name("seasonal_naive_7"),
    }

    eval_frames = get_evaluation_frames(dataset)
    val_df = eval_frames.get("validation")
    if isinstance(val_df, pd.DataFrame) and not val_df.empty and "model_id" in val_df.columns:
        for mid in val_df["model_id"].dropna().unique():
            mid_str = str(mid)
            if mid_str not in options:
                options[mid_str] = friendly_model_name(mid_str)

    return options


def _ensure_model_artifact(
    dataset: str,
    bundle: DataBundle,
    model_id: str,
    run_id: str,
    registry_path: Path,
) -> None:
    """Ensure artifact exists in registry; if missing, fit and persist on-demand."""
    dataset_name = DATASET_NAMES.get(dataset, dataset)
    dest = registry_path / dataset_name / model_id / run_id
    if (dest / "metadata.json").exists() and (dest / "model_registry.joblib").exists():
        return
    try:
        from fnb_forecast.evaluation.pipeline import (
            build_model_factories,
            _build_deploy_features,
            _deployment_features,
        )
        from fnb_forecast.evaluation.uncertainty import ResidualIntervalCalibrator
        from datetime import datetime, timezone
        import platform

        config_root = Path(__file__).parents[3] / "configs"
        factories = build_model_factories(config_root, profile="fast")
        if model_id not in factories:
            return

        deploy_history = bundle.daily_sales.copy()
        deploy_builder = _build_deploy_features(bundle)
        deploy_history_features = _deployment_features(bundle, deploy_builder, deploy_history)
        feature_state = deploy_builder.state_dict()

        reports_root = Path(os.environ.get("FNB_FORECAST_REPORT_ROOT", "artifacts/reports")) / dataset_name
        val_preds_path = reports_root / "validation_predictions.csv"
        if val_preds_path.exists():
            all_val_preds = pd.read_csv(val_preds_path)
            val_df = all_val_preds.loc[all_val_preds["model_id"].eq(model_id)].copy()
        else:
            val_df = pd.DataFrame()

        if not val_df.empty:
            calibrator = ResidualIntervalCalibrator(coverage=0.80).fit(val_df)
            interval_state = calibrator.state_for_json()
        else:
            interval_state = {}

        model = factories[model_id]()
        model.fit(deploy_history_features, deploy_history_features.iloc[0:0].copy())

        checksum = bundle.provenance.checksum or "unknown"
        meta = {
            "config_hash": "on_demand",
            "config_path": "configs/base.yaml",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_checksum": checksum,
            "dataset_name": dataset_name,
            "features": "full",
            "model_id": model_id,
            "platform": platform.platform(),
            "profile": "on_demand",
            "provenance": dataset,
            "python_version": platform.python_version(),
            "run_id": run_id,
            "schema_version": 1,
            "seeds": [20260813],
            "selected": False,
            "train_cutoff": str(pd.to_datetime(deploy_history["date"]).max().date()),
            "validation_wape": 0.0,
        }
        if val_df.empty:
            val_df = pd.DataFrame([{"actual": 1.0, "yhat": 1.0, "role": "validation"}])

        ModelRegistry(registry_path).save(
            dataset_name=dataset_name,
            model_id=model_id,
            run_id=run_id,
            model=model,
            validation_predictions=val_df,
            metadata=meta,
            feature_state=feature_state,
            interval_state=interval_state,
        )
    except Exception:
        pass


def run_forecast(
    origin: pd.Timestamp,
    service_level: float,
    *,
    dataset: str = DEFAULT_DATASET,
    bundle_root: Path | None = None,
    registry_root: Path | None = None,
    model_id: str | None = None,
) -> PlanningOutput:
    """Run the service from the dashboard action without training a model."""
    root = bundle_root or (Path(os.environ.get("FNB_FORECAST_BUNDLE", "data/processed")) / dataset)
    registry = registry_root or Path(os.environ.get("FNB_FORECAST_REGISTRY", "artifacts/registry"))
    bundle = _load_bundle_with_provenance(root, dataset)
    selection = _resolve_selection(dataset, registry)

    if model_id and model_id != "auto":
        selected_model_id = model_id
        selected_run_id = f"{model_id}-validation"
        no_weather_model_id = model_id
        no_weather_run_id = f"{model_id}-validation"
        active_selection = {
            "selected_model_id": selected_model_id,
            "selected_run_id": selected_run_id,
            "no_weather_model_id": no_weather_model_id,
            "no_weather_run_id": no_weather_run_id,
        }
    else:
        selected_model_id = selection["selected_model_id"]
        selected_run_id = selection["selected_run_id"]
        no_weather_model_id = selection.get("no_weather_model_id")
        no_weather_run_id = selection.get("no_weather_run_id")
        active_selection = selection

    _ensure_model_artifact(dataset, bundle, selected_model_id, selected_run_id, registry)

    weather_builder = FeatureBuilder(calendar=bundle.calendar, weather=bundle.weather)
    no_weather_builder = (
        FeatureBuilder(calendar=bundle.calendar, weather=bundle.weather, include_weather=False)
        if no_weather_model_id
        else None
    )
    interval_calibrator = _load_interval_calibrator(
        registry, DATASET_NAMES[dataset], active_selection
    )
    service = ForecastService(
        registry=ModelRegistry(registry),
        feature_builder=weather_builder,
        selected_model_id=selected_model_id,
        selected_run_id=selected_run_id,
        interval_calibrator=interval_calibrator,
        no_weather_feature_builder=no_weather_builder,
        no_weather_model_id=no_weather_model_id,
        no_weather_run_id=no_weather_run_id,
    )
    result = service.run(
        ForecastRequest(
            forecast_origin=pd.Timestamp(origin).normalize(),
            horizon_days=7,
            dataset_name=DATASET_NAMES[dataset],
        ),
        bundle,
        service_level,
    )
    result = enrich_output_with_bundle(result, bundle)
    st.session_state["dashboard_output"] = result
    st.session_state["dashboard_bundle"] = bundle
    st.session_state["selected_dataset"] = dataset
    st.session_state["selected_model_id"] = selected_model_id
    return result


def get_evaluation_frames(dataset: str = DEFAULT_DATASET) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Load optional validation/test reports without making the dashboard brittle."""
    root = Path(os.environ.get("FNB_FORECAST_REPORT_ROOT", "artifacts/reports"))
    candidates = [root / dataset, root / DATASET_NAMES.get(dataset, dataset)]
    names = {
        "validation": ("validation_leaderboard.csv", "leaderboard_validation.csv", "leaderboard.csv"),
        "test": ("test_leaderboard.csv", "leaderboard_test.csv"),
        "leaderboard": ("leaderboard.csv",),
        "ablation": ("ablation_metrics.csv", "ablation.csv"),
        "folds": ("fold_metrics.csv", "folds.csv"),
    }
    result: dict[str, pd.DataFrame | dict[str, Any]] = {
        key: pd.DataFrame() for key in names
    }
    result["metadata"] = {"provenance": DATASET_NAMES.get(dataset, dataset)}
    for directory in candidates:
        if not directory.exists():
            continue
        for key, file_names in names.items():
            for file_name in file_names:
                path = directory / file_name
                if path.exists():
                    try:
                        result[key] = pd.read_csv(path)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        result[key] = pd.DataFrame()
                    break
        report_path = directory / "report.json"
        if report_path.exists():
            try:
                result["metadata"] = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return result
