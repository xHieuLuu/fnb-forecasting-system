"""Versioned model registry with validation-only residual artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fnb_forecast.exceptions import ArtifactCompatibilityError

SCHEMA_VERSION = 1


def config_hash(config: Mapping[str, Any] | Any) -> str:
    """Hash JSON-compatible configuration values canonically."""
    payload = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class ModelRegistry:
    """Persist deployable model runs below ``<root>/<dataset>/<model>/<run>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(
        self,
        *,
        dataset_name: str,
        model_id: str,
        run_id: str,
        model: Any,
        validation_predictions: pd.DataFrame,
        metadata: Mapping[str, Any],
        feature_state: Mapping[str, Any] | None = None,
        interval_state: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write an artifact only when all persisted predictions are validation rows."""
        self._validate_validation_predictions(validation_predictions)
        destination = self.root / dataset_name / model_id / run_id
        destination.mkdir(parents=True, exist_ok=True)
        if model is not None:
            model_path = destination / "model"
            try:
                model.save(model_path)
            except Exception:
                # If model is already saved (e.g. checkpoint loaded from disk where _trainer is None),
                # verify model_path contains artifacts rather than failing.
                if not (model_path.exists() and any(model_path.iterdir())):
                    raise
            try:
                joblib.dump(model, destination / "model_registry.joblib")
            except Exception:
                # Some complex architectures (e.g. PyTorch Forecasting TFT with local classes)
                # cannot be pickled by joblib/pickle, but native model.save() persists everything.
                pass
        validation_predictions.to_parquet(
            destination / "validation_predictions.parquet", index=False
        )
        (destination / "feature_state.json").write_text(
            json.dumps(dict(feature_state or {}), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        (destination / "interval_state.json").write_text(
            json.dumps(dict(interval_state or {}), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        complete_metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "model_id": model_id,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            **dict(metadata),
        }
        (destination / "metadata.json").write_text(
            json.dumps(complete_metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return destination

    def load(
        self,
        dataset_name: str,
        model_id: str,
        run_id: str,
        *,
        expected_config_hash: str | None = None,
    ) -> Any:
        """Load a run and reject incompatible schema or frozen configuration."""
        destination = self.root / dataset_name / model_id / run_id
        metadata_path = destination / "metadata.json"
        if not metadata_path.exists():
            raise ArtifactCompatibilityError(f"artifact metadata missing: {destination}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactCompatibilityError("artifact schema mismatch")
        if expected_config_hash is not None and metadata.get("config_hash") != expected_config_hash:
            raise ArtifactCompatibilityError("artifact config mismatch")
        model_path = destination / "model_registry.joblib"
        if model_path.exists():
            try:
                return joblib.load(model_path)
            except Exception:
                pass
        custom_model_dir = destination / "model"
        if custom_model_dir.exists():
            if "tft" in model_id:
                from fnb_forecast.models.tft import TFTForecastModel
                return TFTForecastModel.load(custom_model_dir)
            if "lstm" in model_id:
                from fnb_forecast.models.lstm import LSTMForecastModel
                return LSTMForecastModel.load(custom_model_dir)
            if "lgb" in model_id:
                from fnb_forecast.models.lightgbm import LightGBMForecastModel
                return LightGBMForecastModel.load(custom_model_dir)
            if "ets" in model_id:
                from fnb_forecast.models.ets import ETSModel
                return ETSModel.load(custom_model_dir)
            if "naive" in model_id:
                from fnb_forecast.models.naive import SeasonalNaiveModel
                return SeasonalNaiveModel.load(custom_model_dir)
        raise ArtifactCompatibilityError("serialized model is missing or incompatible")

    def load_validation_predictions(self, dataset_name: str, model_id: str) -> pd.DataFrame:
        """Read validation rows from all runs of one model, rejecting test leakage."""
        model_root = self.root / dataset_name / model_id
        if not model_root.exists():
            raise ArtifactCompatibilityError(f"model registry entry missing: {model_root}")
        frames: list[pd.DataFrame] = []
        for prediction_path in sorted(model_root.glob("*/validation_predictions.parquet")):
            frame = pd.read_parquet(prediction_path)
            self._validate_validation_predictions(frame)
            frames.append(frame)
        if not frames:
            raise ArtifactCompatibilityError("validation predictions are missing")
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _validate_validation_predictions(frame: pd.DataFrame) -> None:
        required = {"actual", "yhat", "role"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"validation predictions missing columns: {missing}")
        roles = set(frame["role"].dropna().astype(str))
        if roles != {"validation"}:
            raise ValueError("registry accepts validation rows only")
