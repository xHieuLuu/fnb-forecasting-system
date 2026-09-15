"""Validation-only residual prediction intervals."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class ResidualIntervalCalibrator:
    """Calibrate empirical residual intervals without touching held-out test rows."""

    def __init__(self, coverage: float = 0.80, min_group_samples: int = 20) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be between zero and one")
        if min_group_samples < 1:
            raise ValueError("min_group_samples must be positive")
        self.coverage = coverage
        self.min_group_samples = min_group_samples
        self.state: dict[str, object] = {"coverage": coverage, "quantiles": {}}

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> ResidualIntervalCalibrator:
        """Restore a persisted validation-only calibration state."""
        calibrator = cls(coverage=float(state.get("coverage", 0.80)))
        quantiles: dict[tuple[object, int], tuple[float, float]] = {}
        for row in state.get("quantiles", []):
            if not isinstance(row, Mapping):
                continue
            category = row.get("category")
            horizon = int(row["horizon"])
            quantiles[(category, horizon)] = (float(row["q10"]), float(row["q90"]))
        calibrator.state = {"coverage": calibrator.coverage, "quantiles": quantiles}
        return calibrator

    def fit(self, validation_predictions: pd.DataFrame) -> ResidualIntervalCalibrator:
        required = {"actual", "yhat", "role", "horizon"}
        missing = sorted(required.difference(validation_predictions.columns))
        if missing:
            raise ValueError(f"validation_predictions missing columns: {missing}")
        roles = set(validation_predictions["role"].dropna().astype(str))
        if roles != {"validation"}:
            raise ValueError("residual intervals require validation rows only")
        residuals = validation_predictions.copy()
        residuals["residual"] = residuals["actual"] - residuals["yhat"]
        alpha = (1.0 - self.coverage) / 2.0
        quantiles: dict[tuple[object, int], tuple[float, float]] = {}
        horizon_groups = residuals.groupby("horizon", dropna=False, sort=True)
        horizon_quantiles = {
            int(horizon): (
                float(values["residual"].quantile(alpha)),
                float(values["residual"].quantile(1.0 - alpha)),
            )
            for horizon, values in horizon_groups
        }
        for horizon, _ in horizon_groups:
            quantiles[("__horizon__", int(horizon))] = horizon_quantiles[int(horizon)]
        if "category" not in residuals.columns:
            residuals["category"] = "__all__"
        for (category, horizon), values in residuals.groupby(
            ["category", "horizon"], dropna=False, sort=True
        ):
            horizon_key = int(horizon)
            if len(values) >= self.min_group_samples:
                quantiles[(str(category), horizon_key)] = (
                    float(values["residual"].quantile(alpha)),
                    float(values["residual"].quantile(1.0 - alpha)),
                )
            else:
                fallback = horizon_quantiles[horizon_key]
                quantiles[("__horizon__", horizon_key)] = fallback
        self.state = {"coverage": self.coverage, "quantiles": quantiles}
        return self

    def apply(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if not self.state.get("quantiles"):
            raise RuntimeError("ResidualIntervalCalibrator must be fit before apply")
        required = {"yhat", "horizon"}
        missing = sorted(required.difference(predictions.columns))
        if missing:
            raise ValueError(f"predictions missing columns: {missing}")
        result = predictions.copy()
        categories = result.get("category", pd.Series("__all__", index=result.index))
        quantiles = self.state["quantiles"]
        assert isinstance(quantiles, Mapping)
        lowers: list[float] = []
        uppers: list[float] = []
        for index, row in result.iterrows():
            category_key = (str(categories.loc[index]), int(row["horizon"]))
            horizon_key = ("__horizon__", int(row["horizon"]))
            q10, q90 = quantiles.get(category_key, quantiles[horizon_key])
            lower = max(0.0, float(row["yhat"]) + float(q10))
            upper = max(lower, float(row["yhat"]) + float(q90))
            lowers.append(lower)
            uppers.append(upper)
        result["lower"] = np.asarray(lowers, dtype="float64")
        result["upper"] = np.asarray(uppers, dtype="float64")
        return result

    def state_for_json(self) -> dict[str, object]:
        quantiles = self.state.get("quantiles", {})
        assert isinstance(quantiles, Mapping)
        return {
            "coverage": self.coverage,
            "quantiles": [
                {"category": key[0], "horizon": key[1], "q10": value[0], "q90": value[1]}
                for key, value in quantiles.items()
            ],
        }
