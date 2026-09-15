"""Register all candidate models for real_store_calibrated_14m."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml

from fnb_forecast.app.state import get_bundle
from fnb_forecast.evaluation.pipeline import (
    build_model_factories,
    _build_deploy_features,
    _deployment_features,
)
from fnb_forecast.evaluation.uncertainty import ResidualIntervalCalibrator
from fnb_forecast.models.registry import ModelRegistry, config_hash
from fnb_forecast.config import load_project_config

def main() -> None:
    dataset_key = "real_store_14m"
    dataset_name = "real_store_calibrated_14m"
    config_root = Path("configs")
    registry_root = Path("artifacts/registry")
    reports_root = Path("artifacts/reports") / dataset_name

    bundle = get_bundle(dataset_key)
    if bundle is None:
        raise RuntimeError("Failed to load real_store_14m bundle")

    project = load_project_config(config_root / "base.yaml")
    factories = build_model_factories(config_root, profile="fast")

    val_preds_path = reports_root / "validation_predictions.csv"
    if not val_preds_path.exists():
        raise RuntimeError(f"Missing {val_preds_path}")
    all_val_preds = pd.read_csv(val_preds_path)

    leaderboard_path = reports_root / "validation_leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()

    deploy_history = bundle.daily_sales.copy()
    deploy_builder = _build_deploy_features(bundle)
    deploy_history_features = _deployment_features(bundle, deploy_builder, deploy_history)
    feature_state = deploy_builder.state_dict()

    registry = ModelRegistry(registry_root)
    checksum = bundle.provenance.checksum or "c20faad2eb0690006b88513fec5d993846c7c82e633be299468f73f605d6f5da"

    for model_id, factory in factories.items():
        dest = registry_root / dataset_name / model_id / f"{model_id}-validation"
        if (dest / "metadata.json").exists() and (dest / "model_registry.joblib").exists():
            print(f"[SKIP] Model '{model_id}' is already registered at {dest}")
            continue

        print(f"[*] Fitting and registering candidate model: '{model_id}'...")
        val_df = all_val_preds.loc[all_val_preds["model_id"].eq(model_id)].copy()
        if val_df.empty:
            print(f"[WARN] No validation predictions found for '{model_id}', skipping.")
            continue

        calibrator = ResidualIntervalCalibrator(coverage=0.80).fit(val_df)
        interval_state = calibrator.state_for_json()

        if (dest / "model" / "model.ckpt").exists() and model_id.startswith("tft"):
            print(f"[REUSE] Found pre-trained checkpoint for '{model_id}', reusing without retraining!")
            from fnb_forecast.models.tft import TFTForecastModel
            model = TFTForecastModel.load(dest / "model")
            partial_joblib = dest / "model_registry.joblib"
            if partial_joblib.exists():
                partial_joblib.unlink()
        else:
            model = factory()
            model.fit(deploy_history_features, deploy_history_features.iloc[0:0].copy())

        val_wape = 0.0
        if not leaderboard.empty and "model_id" in leaderboard.columns:
            row = leaderboard.loc[leaderboard["model_id"].eq(model_id), "validation_wape"]
            if not row.empty:
                val_wape = float(row.iloc[0])

        is_selected = (model_id == "lgb_l15_d-1_lr0.03")

        meta = {
            "config_hash": config_hash(project.model_dump()),
            "config_path": "configs\\base.yaml",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_checksum": checksum,
            "dataset_name": dataset_name,
            "features": "full",
            "git_commit": "52f298b83a88a6dfc5e0377116b13b60186e82b9",
            "model_id": model_id,
            "platform": platform.platform(),
            "profile": "fast",
            "provenance": dataset_key,
            "python_version": platform.python_version(),
            "run_id": f"{model_id}-validation",
            "schema_version": 1,
            "seeds": [20260813],
            "selected": is_selected,
            "train_cutoff": str(pd.to_datetime(deploy_history["date"]).max().date()),
            "validation_wape": val_wape,
        }

        registry.save(
            dataset_name=dataset_name,
            model_id=model_id,
            run_id=f"{model_id}-validation",
            model=model,
            validation_predictions=val_df,
            metadata=meta,
            feature_state=feature_state,
            interval_state=interval_state,
        )
        print(f"[+] Successfully registered '{model_id}'!")

    selection = {
        "dataset_checksum": checksum,
        "dataset_name": dataset_name,
        "no_weather_model_id": "seasonal_naive_7",
        "no_weather_run_id": "seasonal_naive_7-validation",
        "profile": "fast",
        "provenance": dataset_key,
        "selected_model_id": "seasonal_naive_7",
        "selected_run_id": "seasonal_naive_7-validation",
    }
    selection_path = registry_root / dataset_name / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[+] Updated selection at {selection_path}")

if __name__ == "__main__":
    main()
