import json
from pathlib import Path

import pandas as pd
import pytest

from fnb_forecast.exceptions import ArtifactCompatibilityError
from fnb_forecast.models import SeasonalNaiveModel
from fnb_forecast.models.registry import ModelRegistry, config_hash


def validation_predictions(role: str = "validation") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin": pd.Timestamp("2025-01-01"),
            "target_date": pd.date_range("2025-01-02", periods=2),
            "item_id": ["coffee", "coffee"],
            "yhat": [3.0, 4.0],
            "actual": [4.0, 5.0],
            "model_id": "seasonal_naive_7",
            "fold_name": "validation_01",
            "role": role,
            "dataset_name": "synthetic_vietnam_scenario",
        }
    )


def test_registry_persists_validation_predictions_and_metadata(tmp_path: Path) -> None:
    model = SeasonalNaiveModel()
    history = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=8),
            "item_id": "coffee",
            "quantity": range(8),
        }
    )
    model.fit(history, history.iloc[0:0])
    registry = ModelRegistry(tmp_path)

    registry.save(
        dataset_name="synthetic_vietnam_scenario",
        model_id=model.model_id,
        run_id="run-1",
        model=model,
        validation_predictions=validation_predictions(),
        metadata={
            "dataset_checksum": "abc",
            "config_hash": config_hash({"season_length": 7}),
            "provenance": "synthetic_vietnam_scenario",
        },
        feature_state={"columns": ["quantity"]},
        interval_state={"quantiles": {}},
    )

    loaded = registry.load_validation_predictions("synthetic_vietnam_scenario", model.model_id)
    assert len(loaded) == 2
    assert loaded["role"].eq("validation").all()
    metadata = json.loads(
        (
            tmp_path / "synthetic_vietnam_scenario" / model.model_id / "run-1" / "metadata.json"
        ).read_text()
    )
    assert metadata["schema_version"] == 1


def test_registry_rejects_non_validation_rows(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ValueError, match="validation"):
        registry.save(
            dataset_name="synthetic_vietnam_scenario",
            model_id="seasonal_naive_7",
            run_id="run-1",
            model=None,
            validation_predictions=validation_predictions(role="test"),
            metadata={},
        )


def test_registry_rejects_config_mismatch(tmp_path: Path) -> None:
    model = SeasonalNaiveModel()
    history = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=8), "item_id": "coffee", "quantity": range(8)}
    )
    model.fit(history, history.iloc[0:0])
    registry = ModelRegistry(tmp_path)
    registry.save(
        dataset_name="synthetic_vietnam_scenario",
        model_id=model.model_id,
        run_id="run-1",
        model=model,
        validation_predictions=validation_predictions(),
        metadata={"config_hash": config_hash({"season_length": 7})},
    )

    with pytest.raises(ArtifactCompatibilityError, match="config"):
        registry.load(
            "synthetic_vietnam_scenario",
            model.model_id,
            "run-1",
            expected_config_hash=config_hash({"season_length": 1}),
        )
