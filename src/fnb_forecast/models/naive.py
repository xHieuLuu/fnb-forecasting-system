"""Seasonal-naive statistical baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1


class SeasonalNaiveModel:
    """Forecast each item from its value on the exact seasonal-lag date."""

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self.model_id = f"seasonal_naive_{season_length}"
        self.metadata: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "season_length": season_length,
            "fallback_reason": None,
        }
        self._history: pd.DataFrame | None = None
        self._item_medians: pd.Series | None = None

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        """Store only historical targets; future-known values are never fitted."""
        del known_covariates
        required = {"date", "item_id", "quantity"}
        missing = sorted(required.difference(history.columns))
        if missing:
            raise ValueError(f"history missing columns: {missing}")
        fitted = history.loc[:, ["date", "item_id", "quantity"]].copy()
        fitted["date"] = pd.to_datetime(fitted["date"], errors="raise").dt.normalize()
        fitted["quantity"] = pd.to_numeric(fitted["quantity"], errors="raise")
        if fitted.duplicated(["item_id", "date"]).any():
            raise ValueError("history must have one row per item and date")
        self._history = fitted
        self._item_medians = fitted.groupby("item_id", dropna=False)["quantity"].median()

    def predict(
        self, history: pd.DataFrame, future_covariates: pd.DataFrame
    ) -> pd.DataFrame:
        """Return exact-date seasonal lags, falling back to per-item medians."""
        del history
        if self._history is None or self._item_medians is None:
            raise RuntimeError("SeasonalNaiveModel must be fit before predict")
        required = {"target_date", "item_id"}
        missing = sorted(required.difference(future_covariates.columns))
        if missing:
            raise ValueError(f"future_covariates missing columns: {missing}")

        requested = future_covariates.loc[:, ["target_date", "item_id"]].copy()
        requested["target_date"] = pd.to_datetime(
            requested["target_date"], errors="raise"
        ).dt.normalize()
        if requested.duplicated(["item_id", "target_date"]).any():
            raise ValueError("future_covariates must have one row per item and target date")
        requested["_row_order"] = range(len(requested))
        requested["lag_date"] = requested["target_date"] - pd.Timedelta(
            self.season_length, unit="D"
        )
        lag_values = self._history.rename(
            columns={"date": "lag_date", "quantity": "yhat"}
        )
        result = requested.merge(
            lag_values,
            on=["item_id", "lag_date"],
            how="left",
            validate="many_to_one",
        )
        fallback = result["yhat"].isna()
        if fallback.any():
            result.loc[fallback, "yhat"] = result.loc[fallback, "item_id"].map(
                self._item_medians
            )
            self.metadata["fallback_reason"] = "missing_seasonal_lag"
        else:
            self.metadata["fallback_reason"] = None
        if result["yhat"].isna().any():
            missing_items = sorted(result.loc[result["yhat"].isna(), "item_id"].unique())
            raise ValueError(f"non-null training median unavailable for items: {missing_items}")

        result = result.sort_values("_row_order")
        forecast_origin = result["target_date"].min() - pd.Timedelta(1, unit="D")
        return result.loc[:, ["target_date", "item_id", "yhat"]].assign(
            forecast_origin=forecast_origin,
            model_id=self.model_id,
        )

    def save(self, path: Path) -> None:
        """Persist versioned adapter metadata."""
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
