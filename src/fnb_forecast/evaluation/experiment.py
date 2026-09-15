"""Reproducible experiment selection helpers and rolling-run orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fnb_forecast.contracts import DataBundle, ForecastModel

from .backtest import BacktestRunner
from .metrics import forecast_bias, mae, wape
from .splits import ForecastFold, make_evaluation_splits


@dataclass
class ExperimentResult:
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    leaderboard: pd.DataFrame
    ablation_metrics: pd.DataFrame
    selected_model_id: str
    selected_run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def select_model(leaderboard: pd.DataFrame) -> str:
    """Select by validation WAPE, then complexity and inference cost within 1%."""
    required = {"model_id", "validation_wape"}
    missing = sorted(required.difference(leaderboard.columns))
    if missing:
        raise ValueError(f"leaderboard missing columns: {missing}")
    forbidden = [column for column in leaderboard.columns if "test" in column.lower()]
    if forbidden:
        raise ValueError("model selection must not inspect test metrics")
    board = leaderboard.copy()
    board["validation_wape"] = pd.to_numeric(board["validation_wape"], errors="raise")
    if board.empty or not board["validation_wape"].notna().all():
        raise ValueError("leaderboard must contain finite validation WAPE values")
    minimum = float(board["validation_wape"].min())
    tied = board.loc[board["validation_wape"].le(minimum * 1.01)].copy()
    for column, default in [("parameter_count", float("inf")), ("inference_ms", float("inf"))]:
        if column not in tied:
            tied[column] = default
    selected = tied.sort_values(
        ["parameter_count", "inference_ms", "model_id"], kind="stable"
    ).iloc[0]
    return str(selected["model_id"])


def select_models_by_dataset(leaderboard: pd.DataFrame) -> dict[str, str]:
    """Select each dataset independently, keeping public and synthetic results separate."""
    if "dataset_name" not in leaderboard.columns:
        raise ValueError("leaderboard missing columns: ['dataset_name']")
    return {
        str(dataset_name): select_model(rows.drop(columns="dataset_name"))
        for dataset_name, rows in leaderboard.groupby("dataset_name", sort=True)
    }


class ExperimentRunner:
    """Run validation for all supplied candidates, then test only the winner."""

    def __init__(
        self,
        *,
        backtest_runner: BacktestRunner | None = None,
        model_factories: Mapping[str, Callable[[], ForecastModel]] | None = None,
    ) -> None:
        self.backtest_runner = backtest_runner or BacktestRunner()
        self.model_factories = dict(model_factories or {})

    def run(
        self,
        dataset_name: str,
        bundle: DataBundle,
        config: Any | None = None,
        *,
        validation_folds: Sequence[ForecastFold] | None = None,
        test_folds: Sequence[ForecastFold] | None = None,
    ) -> ExperimentResult:
        """Evaluate candidates on validation before exposing any test fold."""
        del config
        if not self.model_factories:
            raise ValueError("at least one model factory is required")
        dates = pd.DatetimeIndex(pd.to_datetime(bundle.daily_sales["date"]).dt.normalize().unique())
        if validation_folds is None or test_folds is None:
            generated_validation, generated_test = make_evaluation_splits(dates)
            validation_folds = validation_folds or generated_validation
            test_folds = test_folds or generated_test
        validation_parts: list[pd.DataFrame] = []
        leaderboard_rows: list[dict[str, object]] = []
        total_models = len(self.model_factories)
        for idx, (model_id, factory) in enumerate(self.model_factories.items(), start=1):
            print(f"[*] [{idx}/{total_models}] Evaluating candidate model: '{model_id}'...")
            predictions = self.backtest_runner.run(factory, bundle, validation_folds)
            predictions = predictions.assign(
                model_id=model_id,
                dataset_name=dataset_name,
                horizon=(
                    pd.to_datetime(predictions["target_date"])
                    - pd.to_datetime(predictions["forecast_origin"])
                ).dt.days.astype("int64"),
            )
            validation_parts.append(predictions)
            leaderboard_rows.append(self._leaderboard_row(model_id, predictions))
        leaderboard = pd.DataFrame(leaderboard_rows)
        selected_model_id = select_model(leaderboard)
        print(f"[+] Validation complete. Selected best model: '{selected_model_id}'")
        selected_predictions = next(
            frame for frame in validation_parts if frame["model_id"].eq(selected_model_id).all()
        )
        print(f"[*] Running test evaluation for selected model '{selected_model_id}'...")
        test_predictions = self.backtest_runner.run(
            self.model_factories[selected_model_id], bundle, test_folds
        ).assign(model_id=selected_model_id, dataset_name=dataset_name)
        return ExperimentResult(
            validation_predictions=pd.concat(validation_parts, ignore_index=True),
            test_predictions=test_predictions,
            leaderboard=leaderboard,
            ablation_metrics=pd.DataFrame(),
            selected_model_id=selected_model_id,
            selected_run_id=f"{selected_model_id}-validation",
            metadata={
                "dataset_name": dataset_name,
                "selection_uses": "validation_wape_only",
                "selected_validation_rows": len(selected_predictions),
            },
        )

    @staticmethod
    def _leaderboard_row(model_id: str, predictions: pd.DataFrame) -> dict[str, object]:
        actual = predictions["actual"].to_numpy(dtype="float64")
        predicted = predictions["yhat"].to_numpy(dtype="float64")
        return {
            "model_id": model_id,
            "validation_wape": wape(actual, predicted),
            "validation_mae": mae(actual, predicted),
            "validation_bias": forecast_bias(actual, predicted),
            "parameter_count": float("inf"),
            "inference_ms": float("inf"),
        }
