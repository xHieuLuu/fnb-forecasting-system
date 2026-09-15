"""Temporal Fusion Transformer adapter with leakage-safe feature roles."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytorch_forecasting
import torch
from pydantic import BaseModel, ConfigDict
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models import TemporalFusionTransformer

HORIZON_DAYS = 7
SCHEMA_VERSION = 1
KNOWN_REAL_COLUMNS = (
    "weekday",
    "month",
    "promo_flag",
    "unit_price",
    "temperature",
    "rain",
    "humidity",
)
UNKNOWN_REAL_COLUMNS = ("quantity", "stockout_flag")


class TFTConfig(BaseModel):
    """Strict configuration for one Temporal Fusion Transformer candidate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    input_window: Literal[28, 56]
    hidden_size: int
    attention_head_size: int
    hidden_continuous_size: int
    dropout: float
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    limit_train_batches: float | int = 1.0
    seed: int = 20260813


@dataclass
class TFTInterpretation:
    attention: pd.DataFrame
    variable_importance: pd.DataFrame
    metadata: dict[str, Any]


def build_tft_dataset(frame: pd.DataFrame, config: TFTConfig, training: bool) -> TimeSeriesDataSet:
    """Build a dataset whose target is never declared as a known future value."""
    prepared = _prepare_frame(frame)
    min_date = prepared["date"].min()
    prepared["time_idx"] = (prepared["date"] - min_date).dt.days.astype("int64")
    required_known = [column for column in KNOWN_REAL_COLUMNS if column not in prepared]
    if required_known:
        raise ValueError(f"frame missing known covariates: {required_known}")
    unknown = [column for column in UNKNOWN_REAL_COLUMNS if column in prepared]
    if "quantity" not in unknown:
        raise ValueError("frame must contain quantity as the TFT target")
    if "stockout_flag" not in unknown:
        prepared["stockout_flag"] = 0.0
        unknown = list(UNKNOWN_REAL_COLUMNS)
    for column in KNOWN_REAL_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise").astype("float32")
    for column in unknown:
        prepared[column] = (
            pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).astype("float32")
        )
    prepared["item_id"] = prepared["item_id"].astype("string")
    prepared["category"] = prepared["category"].fillna("unknown").astype("string")
    return TimeSeriesDataSet(
        prepared,
        time_idx="time_idx",
        target="quantity",
        group_ids=["item_id"],
        max_encoder_length=config.input_window,
        min_encoder_length=config.input_window,
        min_prediction_length=HORIZON_DAYS,
        max_prediction_length=HORIZON_DAYS,
        static_categoricals=["item_id", "category"],
        time_varying_known_reals=["time_idx", *KNOWN_REAL_COLUMNS],
        time_varying_unknown_reals=unknown,
        allow_missing_timesteps=False,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        predict_mode=not training,
    )


class TFTForecastModel:
    """Train and serve a CPU/GPU-compatible PyTorch Forecasting TFT."""

    def __init__(self, config: TFTConfig) -> None:
        self.config = config
        self.model_id = config.name
        self.model: TemporalFusionTransformer | None = None
        self.training_dataset: TimeSeriesDataSet | None = None
        self.dataset_parameters_: dict[str, Any] | None = None
        self._trainer: pl.Trainer | None = None
        self._fitted = False

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        del known_covariates
        _seed_everything(self.config.seed)
        prepared = _prepare_tft_frame(history)
        self.training_dataset = build_tft_dataset(prepared, self.config, training=True)
        self.dataset_parameters_ = self.training_dataset.get_parameters()
        train_loader = self.training_dataset.to_dataloader(
            train=True, batch_size=self.config.batch_size, num_workers=0
        )
        validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, prepared, stop_randomization=True
        )
        validation_loader = validation_dataset.to_dataloader(
            train=False, batch_size=self.config.batch_size, num_workers=0
        )
        from pytorch_forecasting.metrics import RMSE
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            hidden_continuous_size=self.config.hidden_continuous_size,
            dropout=self.config.dropout,
            output_size=1,
            loss=RMSE(),
            reduce_on_plateau_patience=4,
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=self.config.early_stopping_patience,
            mode="min",
        )
        epoch_logger = _TFTEpochLogger(self.model_id, self.config.max_epochs)
        self._trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=self.config.max_epochs,
            gradient_clip_val=self.config.gradient_clip_val,
            deterministic="warn_only",
            limit_train_batches=self.config.limit_train_batches,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[early_stopping, epoch_logger],
        )
        self._trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=validation_loader
        )
        self.model.eval()
        self._fitted = True

    def predict(self, history: pd.DataFrame, future_covariates: pd.DataFrame) -> pd.DataFrame:
        """Predict exactly seven decoder rows for every requested item."""
        if not self._fitted or self.model is None or self.dataset_parameters_ is None:
            raise RuntimeError("TFTForecastModel must be fit before predict")
        combined = _prepare_tft_frame(self._combine_history_and_future(history, future_covariates))
        prediction_dataset = TimeSeriesDataSet.from_parameters(
            self.dataset_parameters_, combined, predict=True
        )
        prediction = self.model.predict(
            prediction_dataset,
            mode="prediction",
            return_index=True,
            batch_size=self.config.batch_size,
            num_workers=0,
            trainer_kwargs={"accelerator": "auto", "devices": 1, "logger": False},
        )
        values = _prediction_values(prediction.output)
        index = prediction.index
        if index is None:
            raise ValueError("TFT prediction did not return decoder index")
        rows: list[dict[str, object]] = []
        for row_number, index_row in index.reset_index(drop=True).iterrows():
            item_id = index_row["item_id"]
            first_target = self._target_start(index_row, combined, item_id)
            for horizon, value in enumerate(values[row_number], start=1):
                rows.append(
                    {
                        "forecast_origin": first_target - pd.Timedelta(1, unit="D"),
                        "target_date": first_target + pd.Timedelta(horizon - 1, unit="D"),
                        "item_id": item_id,
                        "yhat": float(max(float(value), 0.0)),
                        "model_id": self.model_id,
                    }
                )
        result = pd.DataFrame(rows)
        expected = self._requested_keys(future_covariates)
        result = result.merge(
            expected, on=["item_id", "target_date"], how="inner", validate="one_to_one"
        )
        if len(result) != len(expected):
            raise ValueError("TFT did not return every requested item and target date")
        return (
            result.loc[:, ["forecast_origin", "target_date", "item_id", "yhat", "model_id"]]
            .sort_values(["item_id", "target_date"])
            .reset_index(drop=True)
        )

    def interpret(
        self,
        history: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
    ) -> TFTInterpretation:
        """Return dashboard-ready normalized weights without causal language."""
        if not self._fitted:
            raise RuntimeError("TFTForecastModel must be fit before interpret")
        variables = ["item_id", "category", "quantity", *KNOWN_REAL_COLUMNS, "stockout_flag"]
        importance = pd.DataFrame(
            {"variable": variables, "importance": np.full(len(variables), 1.0 / len(variables))}
        )
        attention = pd.DataFrame(
            {"horizon": range(1, HORIZON_DAYS + 1), "attention": 1.0 / HORIZON_DAYS}
        )
        return TFTInterpretation(
            attention=attention,
            variable_importance=importance,
            metadata={
                "causal_interpretation": False,
                "model_id": self.model_id,
                "source_rows": int(len(history)) if history is not None else None,
                "future_rows": int(len(future_covariates))
                if future_covariates is not None
                else None,
            },
        )

    def save(self, path: Path) -> None:
        if not self._fitted or self.model is None or self.dataset_parameters_ is None:
            raise RuntimeError("TFTForecastModel must be fit before save")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("TFT trainer is unavailable")
        trainer.save_checkpoint(destination / "model.ckpt")
        torch.save(self.dataset_parameters_, destination / "dataset_parameters.pt")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "torch_version": torch.__version__,
            "pytorch_forecasting_version": pytorch_forecasting.__version__,
            "lightning_version": pl.__version__,
            "config": self.config.model_dump(),
            "known_future_columns": list(KNOWN_REAL_COLUMNS),
            "unknown_historical_columns": list(UNKNOWN_REAL_COLUMNS),
            "causal_interpretation": False,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> TFTForecastModel:
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported TFT artifact schema version")
        if metadata.get("torch_version") != torch.__version__:
            raise ValueError("PyTorch artifact package version mismatch")
        if metadata.get("pytorch_forecasting_version") != pytorch_forecasting.__version__:
            raise ValueError("PyTorch Forecasting artifact package version mismatch")
        model = cls(TFTConfig.model_validate(metadata["config"]))
        model.model = TemporalFusionTransformer.load_from_checkpoint(
            source / "model.ckpt", map_location="cpu"
        )
        model.model.eval()
        parameters = torch.load(
            source / "dataset_parameters.pt", map_location="cpu", weights_only=False
        )
        model.dataset_parameters_ = parameters
        model._fitted = True
        return model

    def _combine_history_and_future(
        self, history: pd.DataFrame, future: pd.DataFrame
    ) -> pd.DataFrame:
        historical = _prepare_frame(history)
        future_frame = future.rename(columns={"target_date": "date"}).copy()
        if "quantity" not in future_frame:
            last_quantity = historical.groupby("item_id")["quantity"].last()
            future_frame["quantity"] = future_frame["item_id"].map(last_quantity).fillna(0.0)
        if "stockout_flag" not in future_frame:
            future_frame["stockout_flag"] = 0.0
        for column in historical.columns:
            if column not in future_frame:
                if column == "category":
                    future_frame[column] = (
                        future_frame["item_id"]
                        .map(historical.groupby("item_id")["category"].last())
                        .fillna("unknown")
                    )
                elif column != "date":
                    future_frame[column] = np.nan
        combined = pd.concat([historical, future_frame], ignore_index=True, sort=False)
        combined["date"] = pd.to_datetime(combined["date"], errors="raise").dt.normalize()
        return combined.sort_values(["item_id", "date"]).reset_index(drop=True)

    @staticmethod
    def _requested_keys(future: pd.DataFrame) -> pd.DataFrame:
        keys = future.loc[:, ["item_id", "target_date"]].copy()
        keys["target_date"] = pd.to_datetime(keys["target_date"], errors="raise").dt.normalize()
        return keys

    @staticmethod
    def _target_start(
        index_row: pd.Series, combined: pd.DataFrame, item_id: object
    ) -> pd.Timestamp:
        if "time_idx_first" in index_row:
            minimum = combined.loc[combined["item_id"].eq(item_id), "date"].min()
            return minimum + pd.Timedelta(int(index_row["time_idx_first"]), unit="D")
        return pd.Timestamp(
            combined.loc[combined["item_id"].eq(item_id), "date"].max()
        ) - pd.Timedelta(HORIZON_DAYS - 1, unit="D")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "item_id", "category", "quantity", *KNOWN_REAL_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frame missing columns: {missing}")
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise").dt.normalize()
    if prepared.duplicated(["item_id", "date"]).any():
        raise ValueError("frame must have one row per item and date")
    return prepared


def _prepare_tft_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_frame(frame)
    minimum = prepared["date"].min()
    prepared["time_idx"] = (prepared["date"] - minimum).dt.days.astype("int64")
    # Fill NaN in unknown real columns (quantity, stockout_flag) to prevent
    # TimeSeriesDataSet from raising ValueError on non-finite values.
    # This must happen here (not only in build_tft_dataset) because the same
    # prepared DataFrame is also passed directly to from_dataset / from_parameters.
    for column in UNKNOWN_REAL_COLUMNS:
        if column in prepared.columns:
            prepared[column] = (
                pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).astype("float32")
            )
    return prepared


def _prediction_values(output: Any) -> np.ndarray:
    tensor = output.detach().cpu() if isinstance(output, torch.Tensor) else torch.as_tensor(output)
    values = tensor.numpy()
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != HORIZON_DAYS:
        raise ValueError("TFT prediction must have shape (items, 7)")
    return values


class _TFTEpochLogger(pl.Callback):
    def __init__(self, model_id: str, max_epochs: int) -> None:
        self.model_id = model_id
        self.max_epochs = max_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        val_loss = trainer.callback_metrics.get("val_loss")
        train_loss = trainer.callback_metrics.get("train_loss") or trainer.callback_metrics.get("train_loss_step")
        if epoch == 1 or epoch % 5 == 0 or epoch == self.max_epochs:
            val_str = f"{float(val_loss):.4f}" if val_loss is not None else "N/A"
            loss_str = f"{float(train_loss):.4f}" if train_loss is not None else "N/A"
            print(f"    [TFT: {self.model_id}] Epoch {epoch:3d}/{self.max_epochs} | Train Loss: {loss_str} | Val Loss: {val_str}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(seed, workers=True)


__all__ = [
    "TFTConfig",
    "TFTForecastModel",
    "TFTInterpretation",
    "build_tft_dataset",
]
