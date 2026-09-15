import pandas as pd
import pytest

from fnb_forecast.evaluation.experiment import (
    ExperimentResult,
    select_model,
    select_models_by_dataset,
)


def test_selector_prefers_simpler_model_inside_one_percent_tie() -> None:
    board = pd.DataFrame(
        {
            "model_id": ["lstm", "lightgbm"],
            "validation_wape": [0.199, 0.200],
            "parameter_count": [50_000, 500],
            "inference_ms": [30.0, 4.0],
        }
    )

    assert select_model(board) == "lightgbm"


def test_selector_rejects_test_metrics() -> None:
    board = pd.DataFrame(
        {
            "model_id": ["naive"],
            "validation_wape": [0.2],
            "test_wape": [0.1],
        }
    )

    with pytest.raises(ValueError, match="test"):
        select_model(board)


def test_selector_is_run_independently_per_dataset() -> None:
    board = pd.DataFrame(
        {
            "dataset_name": [
                "m5_public_benchmark",
                "m5_public_benchmark",
                "synthetic_vietnam_scenario",
                "synthetic_vietnam_scenario",
            ],
            "model_id": ["ets", "lstm", "ets", "lstm"],
            "validation_wape": [0.10, 0.20, 0.20, 0.10],
            "parameter_count": [10, 100, 10, 100],
            "inference_ms": [1.0, 2.0, 1.0, 2.0],
        }
    )

    assert select_models_by_dataset(board) == {
        "m5_public_benchmark": "ets",
        "synthetic_vietnam_scenario": "lstm",
    }


def test_experiment_result_has_separate_validation_and_test_frames() -> None:
    result = ExperimentResult(
        validation_predictions=pd.DataFrame({"role": ["validation"]}),
        test_predictions=pd.DataFrame({"role": ["test"]}),
        leaderboard=pd.DataFrame(),
        ablation_metrics=pd.DataFrame(),
        selected_model_id="ets",
        selected_run_id="run-1",
        metadata={"dataset_name": "synthetic_vietnam_scenario"},
    )

    assert result.validation_predictions["role"].eq("validation").all()
    assert result.test_predictions["role"].eq("test").all()
