"""StatsForecast AutoETS adapter."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import statsforecast
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

SCHEMA_VERSION = 1
HORIZON_DAYS = 7


class ETSModel:
    """Fit one AutoETS model per item and forecast the requested seven dates."""

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self.model_id = f"auto_ets_{season_length}"
        self.metadata: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "season_length": season_length,
            "statsforecast_version": statsforecast.__version__,
            "statsforecast_prediction_api": "fit_predict",
        }
        self._statsforecast: StatsForecast | None = None
        self._item_ids: set[object] = set()
        self._last_training_dates: pd.Series | None = None

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        """Fit AutoETS from historical targets only."""
        del known_covariates
        required = {"date", "item_id", "quantity"}
        missing = sorted(required.difference(history.columns))
        if missing:
            raise ValueError(f"history missing columns: {missing}")
        training = history.loc[:, ["item_id", "date", "quantity"]].rename(
            columns={"item_id": "unique_id", "date": "ds", "quantity": "y"}
        )
        training["ds"] = pd.to_datetime(training["ds"], errors="raise").dt.normalize()
        training["y"] = pd.to_numeric(training["y"], errors="raise")
        if training.duplicated(["unique_id", "ds"]).any():
            raise ValueError("history must have one row per item and date")
        training = training.dropna(subset=["y"])
        if training.empty:
            raise ValueError("history must contain non-null targets")

        self._statsforecast = StatsForecast(
            models=[AutoETS(season_length=self.season_length, model="ZZZ")],
            freq="D",
            n_jobs=1,
        ).fit(training)
        self._item_ids = set(training["unique_id"].unique())
        self._last_training_dates = training.groupby("unique_id", dropna=False)["ds"].max()

    def predict(
        self, history: pd.DataFrame, future_covariates: pd.DataFrame
    ) -> pd.DataFrame:
        """Forecast seven steps and map them to the future frame's item/date keys."""
        if self._statsforecast is None or self._last_training_dates is None:
            raise RuntimeError("ETSModel must be fit before predict")
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
        requested_items = set(requested["item_id"].unique())
        if requested_items != self._item_ids:
            raise ValueError("future_covariates items must match fitted history items")
        counts = requested.groupby("item_id", dropna=False)["target_date"].nunique()
        if not counts.eq(HORIZON_DAYS).all():
            raise ValueError("each item must contain exact seven target dates")

        last_training_dates = requested["item_id"].map(self._last_training_dates)
        if (
            last_training_dates.isna().any()
            or (requested["target_date"] - last_training_dates).dt.days.le(0).any()
        ):
            if (
                history is not None
                and not history.empty
                and {"date", "item_id", "quantity"}.issubset(history.columns)
            ):
                self.fit(history, future_covariates)
                last_training_dates = requested["item_id"].map(self._last_training_dates)

        if last_training_dates.isna().any():
            raise ValueError("last non-null training date unavailable for a requested item")
        forecast_steps = (requested["target_date"] - last_training_dates).dt.days
        if forecast_steps.le(0).any():
            raise ValueError("requested target dates must follow each item's fitted history")
        forecast_horizon = int(forecast_steps.max())

        forecast = self._statsforecast.predict(h=forecast_horizon).rename(
            columns={"unique_id": "item_id"}
        )
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="raise").dt.normalize()
        forecast["AutoETS"] = pd.to_numeric(
            forecast["AutoETS"], errors="raise"
        ).clip(lower=0.0)
        result = requested.merge(
            forecast.loc[:, ["item_id", "ds", "AutoETS"]],
            left_on=["item_id", "target_date"],
            right_on=["item_id", "ds"],
            how="left",
            validate="one_to_one",
        )
        if result["AutoETS"].isna().any():
            raise ValueError("AutoETS did not return every requested item and target date")
        result = result.rename(columns={"AutoETS": "yhat"})
        result = result.sort_values(["item_id", "target_date"]).reset_index(drop=True)
        forecast_origin = result["target_date"].min() - pd.Timedelta(1, unit="D")
        return result.loc[:, ["target_date", "item_id", "yhat"]].assign(
            forecast_origin=forecast_origin,
            model_id=self.model_id,
        )

    def save(self, path: Path) -> None:
        """Persist versioned metadata and the fitted StatsForecast object."""
        if self._statsforecast is None:
            raise RuntimeError("ETSModel must be fit before save")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        joblib.dump(self._statsforecast, destination / "model.joblib")
