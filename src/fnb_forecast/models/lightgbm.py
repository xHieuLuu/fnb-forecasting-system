"""Leakage-safe global LightGBM model for seven forecast horizons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict

HORIZON_DAYS = 7
ARTIFACT_SCHEMA_VERSION = 1
TABLE_METADATA_COLUMNS = {"forecast_origin", "target_date", "target"}


class LightGBMConfig(BaseModel):
    """Strict configuration for one candidate global model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    seed: int


TARGET_DATE_FEATURES = {
    "weekday",
    "is_weekend",
    "holiday_name",
    "days_to_tet",
    "days_after_tet",
    "store_open",
    "month",
    "temperature",
    "rain",
    "humidity",
    "unit_price",
    "promo_flag",
}


def build_horizon_training_table(
    history: pd.DataFrame, max_horizon: int = HORIZON_DAYS
) -> pd.DataFrame:
    """Copy origin features and join target-date calendar/weather features for each horizon."""
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")
    required = {"date", "item_id", "quantity"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"history missing columns: {missing}")

    supplied = history.copy()
    supplied["date"] = pd.to_datetime(supplied["date"], errors="raise").dt.normalize()
    if supplied.duplicated(["item_id", "date"]).any():
        raise ValueError("history must have one row per item and date")

    target_cols = [c for c in TARGET_DATE_FEATURES if c in supplied.columns]
    
    # We want to keep some origin features for context
    origin_context = {"weekday", "is_weekend", "holiday_name", "month", "store_open"}
    origin_cols_to_keep = [
        c for c in supplied.columns 
        if c not in {"target_date", "date", "item_id"} 
        and (c not in target_cols or c in origin_context)
    ]

    origins = supplied.loc[:, ["item_id", "date", *origin_cols_to_keep]].rename(
        columns={"date": "forecast_origin"}
    )
    rename_map = {c: f"origin_{c}" for c in origin_context if c in origins.columns}
    origins = origins.rename(columns=rename_map)
    targets = supplied.loc[:, ["item_id", "date", "quantity", *target_cols]].rename(
        columns={"date": "target_date", "quantity": "target"}
    )
    rows: list[pd.DataFrame] = []
    for horizon in range(1, max_horizon + 1):
        horizon_rows = origins.copy()
        horizon_rows["horizon"] = horizon
        horizon_rows["target_date"] = horizon_rows["forecast_origin"] + pd.Timedelta(
            horizon, unit="D"
        )
        rows.append(
            horizon_rows.merge(
                targets,
                on=["item_id", "target_date"],
                how="inner",
                validate="many_to_one",
            )
        )
    table = pd.concat(rows, ignore_index=True)
    complete = table.groupby(["item_id", "forecast_origin"], dropna=False)[
        "target"
    ].transform(lambda values: len(values) == max_horizon and values.notna().all())
    return table.loc[complete].sort_values(
        ["item_id", "forecast_origin", "horizon"]
    ).reset_index(drop=True)


class LightGBMForecastModel:
    """Fit one deterministic regressor across every item and horizon."""

    def __init__(self, config: LightGBMConfig) -> None:
        self.config = config
        self.model_id = config.name
        self.estimator = LGBMRegressor(
            objective="regression",
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            subsample=config.subsample,
            subsample_freq=config.subsample_freq,
            colsample_bytree=config.colsample_bytree,
            random_state=config.seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1,
        )
        self.category_encoders_: dict[str, dict[str, int]] = {}
        self.feature_order_: list[str] = []
        self.training_checksum_: str | None = None
        self._fitted = False

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        """Learn encoders and estimator state solely from supplied history."""
        del known_covariates
        table = build_horizon_training_table(history, max_horizon=HORIZON_DAYS)
        if table.empty:
            raise ValueError("history must contain at least one complete seven-day target window")
        target = pd.to_numeric(table["target"], errors="raise").astype("float64")
        if not np.isfinite(target).all():
            raise ValueError("training targets must be finite")
        self.feature_order_ = [
            column for column in table.columns if column not in TABLE_METADATA_COLUMNS
        ]
        features = table.loc[:, self.feature_order_]
        self.category_encoders_ = self._fit_category_encoders(features)
        self.estimator.fit(self._encode_features(features), target)
        self.training_checksum_ = self._training_checksum(history)
        self._fitted = True

    def predict(
        self, history: pd.DataFrame, future_covariates: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict exact seven-day keys from matching as-of-origin features and target-date covariates."""
        if not self._fitted:
            raise RuntimeError("LightGBMForecastModel must be fit before predict")
        requested = self._validate_future_keys(future_covariates)
        forecast_origin = requested["target_date"].min() - pd.Timedelta(1, unit="D")

        target_features_present = set(self.feature_order_).intersection(TARGET_DATE_FEATURES)
        origin_features_needed = set(self.feature_order_).difference(
            TARGET_DATE_FEATURES | {"horizon"}
        )

        origins = history.copy()
        origins["date"] = pd.to_datetime(origins["date"], errors="raise").dt.normalize()
        origins = origins.loc[origins["date"].eq(forecast_origin)].drop(
            columns="target_date", errors="ignore"
        )
        missing_target = sorted(target_features_present.difference(future_covariates.columns))
        if missing_target:
            raise ValueError(f"future_covariates missing required target features: {missing_target}")

        origin_context = {"weekday", "is_weekend", "holiday_name", "month", "store_open"}
        rename_map = {c: f"origin_{c}" for c in origin_context if c in origins.columns}
        origins = origins.rename(columns=rename_map)

        missing_origin = sorted(origin_features_needed.difference(origins.columns))
        if missing_origin:
            raise ValueError(f"history missing required model features: {missing_origin}")
            
        origin_cols = ["item_id", *sorted(origin_features_needed - {"item_id"})]
        origins = origins.loc[:, origin_cols]
        if origins.duplicated("item_id").any():
            raise ValueError("history must have one origin row per item")
        expected_items = set(requested["item_id"])
        if set(origins["item_id"]) != expected_items:
            raise ValueError(
                "history must contain the forecast-origin row for every requested item"
            )

        target_covs = future_covariates.copy()
        target_covs["target_date"] = pd.to_datetime(target_covs["target_date"]).dt.normalize()
        cov_cols = ["target_date", "item_id", *sorted(target_features_present)]
        target_covs = target_covs.loc[:, cov_cols].drop_duplicates(["target_date", "item_id"])

        prediction_rows = requested.merge(
            target_covs,
            on=["target_date", "item_id"],
            how="left",
            validate="many_to_one",
        ).merge(
            origins,
            on="item_id",
            how="left",
            validate="many_to_one",
        )
        prediction_rows["horizon"] = (
            prediction_rows["target_date"] - forecast_origin
        ).dt.days
        raw = self.estimator.predict(
            self._encode_features(prediction_rows.loc[:, self.feature_order_])
        )
        yhat = np.maximum(np.asarray(raw, dtype="float64"), 0.0)
        if not np.isfinite(yhat).all():
            raise ValueError("LightGBM produced non-finite forecasts")
        result = requested.assign(
            forecast_origin=forecast_origin,
            yhat=yhat,
            model_id=self.model_id,
        )
        return result.loc[
            :, ["forecast_origin", "target_date", "item_id", "yhat", "model_id"]
        ]

    def save(self, path: Path) -> None:
        """Persist fitted model state without retaining training rows or targets."""
        if not self._fitted or self.training_checksum_ is None:
            raise RuntimeError("LightGBMForecastModel must be fit before save")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "package_version": lightgbm.__version__,
            "training_checksum": self.training_checksum_,
            "config": self.config.model_dump(),
            "feature_order": self.feature_order_,
            "category_encoders": self.category_encoders_,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        joblib.dump(self.estimator, destination / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> LightGBMForecastModel:
        """Restore a fitted model and its exact feature compatibility contract."""
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported LightGBM artifact schema version")
        if metadata.get("package_version") != lightgbm.__version__:
            raise ValueError("LightGBM artifact package version mismatch")
        model = cls(LightGBMConfig.model_validate(metadata["config"]))
        model.estimator = joblib.load(source / "model.joblib")
        model.feature_order_ = [str(column) for column in metadata["feature_order"]]
        model.category_encoders_ = {
            str(column): {str(value): int(code) for value, code in mapping.items()}
            for column, mapping in metadata["category_encoders"].items()
        }
        model.training_checksum_ = str(metadata["training_checksum"])
        model._fitted = True
        return model

    @staticmethod
    def _validate_future_keys(future_covariates: pd.DataFrame) -> pd.DataFrame:
        required = {"target_date", "item_id"}
        missing = sorted(required.difference(future_covariates.columns))
        if missing:
            raise ValueError(f"future_covariates missing columns: {missing}")
        requested = future_covariates.loc[:, ["target_date", "item_id"]].copy()
        requested["target_date"] = pd.to_datetime(
            requested["target_date"], errors="raise"
        ).dt.normalize()
        if requested.duplicated(["item_id", "target_date"]).any():
            raise ValueError("future_covariates must have unique item and target-date keys")
        expected_dates = pd.date_range(requested["target_date"].min(), periods=HORIZON_DAYS)
        if set(requested["target_date"]) != set(expected_dates):
            raise ValueError("future_covariates must contain exact seven target dates")
        for _, item_rows in requested.groupby("item_id", dropna=False):
            if set(item_rows["target_date"]) != set(expected_dates):
                raise ValueError("each item must contain exact seven target dates")
        return requested.sort_values(["item_id", "target_date"]).reset_index(drop=True)

    @staticmethod
    def _fit_category_encoders(features: pd.DataFrame) -> dict[str, dict[str, int]]:
        return {
            column: {
                value: code
                for code, value in enumerate(
                    sorted(str(value) for value in features[column].dropna().unique())
                )
            }
            for column in features.columns
            if not is_numeric_dtype(features[column].dtype)
        }

    def _encode_features(self, features: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=features.index)
        for column in self.feature_order_:
            if column in self.category_encoders_:
                encoded[column] = (
                    features[column]
                    .astype("string")
                    .map(self.category_encoders_[column])
                    .fillna(-1)
                    .astype("int32")
                )
            else:
                encoded[column] = pd.to_numeric(features[column], errors="raise")
        return encoded

    @staticmethod
    def _training_checksum(history: pd.DataFrame) -> str:
        canonical = history.sort_index(axis="columns").reset_index(drop=True)
        row_hashes = pd.util.hash_pandas_object(canonical, index=False, categorize=True)
        return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()
