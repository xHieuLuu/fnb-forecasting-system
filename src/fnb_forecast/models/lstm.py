"""Global direct seven-output LSTM forecast adapter."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

HORIZON_DAYS = 7
SCHEMA_VERSION = 1
_DERIVED_PREFIXES = ("lag_", "rolling_")


class LSTMConfig(BaseModel):
    """Strict configuration for one global direct LSTM candidate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    input_window: Literal[28, 56]
    hidden_size: int
    num_layers: int = 2
    dropout: float
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    seed: int = 20260813


@dataclass(frozen=True)
class LSTMWindowSample:
    history_numeric: Tensor
    future_known: Tensor
    item_id: int
    category_id: int
    target: Tensor
    forecast_origin: pd.Timestamp
    target_dates: pd.DatetimeIndex


@dataclass(frozen=True)
class LSTMBatch:
    history_numeric: Tensor
    future_known: Tensor
    item_id: Tensor
    category_id: Tensor
    target: Tensor
    forecast_origins: tuple[pd.Timestamp, ...]
    target_dates: tuple[pd.DatetimeIndex, ...]


class LSTMWindowDataset(Dataset[LSTMWindowSample]):
    """Create leakage-safe rolling windows from a long item/date frame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        input_window: int,
        horizon: int = HORIZON_DAYS,
        numeric_stats: Mapping[str, Mapping[str, float]] | None = None,
        item_encoder: Mapping[str, int] | None = None,
        category_encoder: Mapping[str, int] | None = None,
        item_scales: Mapping[str, float] | None = None,
        history_columns: Sequence[str] | None = None,
        future_columns: Sequence[str] | None = None,
        require_target: bool = True,
    ) -> None:
        if input_window not in (28, 56):
            raise ValueError("input_window must be 28 or 56")
        if horizon != HORIZON_DAYS:
            raise ValueError("horizon must be 7")
        self.input_window = input_window
        self.horizon = horizon
        self.require_target = require_target
        self._frame = self._prepare_frame(frame)
        self.item_encoder = dict(item_encoder or self._build_encoder(self._frame["item_id"]))
        categories = self._frame["category"]
        self.category_encoder = dict(category_encoder or self._build_encoder(categories))
        self.item_scales = dict(item_scales) if item_scales is not None else {}
        self.history_columns = list(history_columns or self._infer_history_columns(self._frame))
        self.future_columns = list(future_columns or self._infer_future_columns(self._frame))
        self.numeric_stats = {
            column: {"mean": float(values["mean"]), "scale": float(values["scale"])}
            for column, values in (numeric_stats or {}).items()
        }
        self.samples = self._make_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LSTMWindowSample:
        return self.samples[index]

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted({"date", "item_id", "quantity"}.difference(frame.columns))
        if missing:
            raise ValueError(f"frame missing columns: {missing}")
        prepared = frame.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise").dt.normalize()
        if prepared.duplicated(["item_id", "date"]).any():
            raise ValueError("frame must have one row per item and date")
        if "category" not in prepared.columns:
            prepared["category"] = "unknown"
        return prepared.sort_values(["item_id", "date"]).reset_index(drop=True)

    @staticmethod
    def _build_encoder(values: pd.Series) -> dict[str, int]:
        return {
            value: index
            for index, value in enumerate(sorted(str(value) for value in values.dropna().unique()))
        }

    @classmethod
    def _infer_history_columns(cls, frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.columns
            if column not in {"date", "target_date", "item_id", "category"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]

    @classmethod
    def _infer_future_columns(cls, frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in cls._infer_history_columns(frame)
            if column != "quantity" and not column.startswith(_DERIVED_PREFIXES)
        ]

    def _make_samples(self) -> list[LSTMWindowSample]:
        samples: list[LSTMWindowSample] = []
        for item_value, item_rows in self._frame.groupby("item_id", sort=True, dropna=False):
            item_rows = item_rows.sort_values("date").reset_index(drop=True)
            for origin_index in range(self.input_window - 1, len(item_rows) - self.horizon):
                history_rows = item_rows.iloc[
                    origin_index - self.input_window + 1 : origin_index + 1
                ]
                target_rows = item_rows.iloc[origin_index + 1 : origin_index + self.horizon + 1]
                origin_row = item_rows.iloc[origin_index]
                if not bool(origin_row.get("store_open", True)):
                    continue
                if not self._is_daily(history_rows["date"]) or not self._is_daily(
                    pd.concat([history_rows["date"].tail(1), target_rows["date"]])
                ):
                    continue
                target = pd.to_numeric(target_rows["quantity"], errors="coerce")
                if self.require_target and target.isna().any():
                    continue
                item_scale = self.item_scales.get(str(item_value), 1.0)
                samples.append(
                    LSTMWindowSample(
                        history_numeric=self._numeric_tensor(history_rows, self.history_columns),
                        future_known=self._numeric_tensor(target_rows, self.future_columns),
                        item_id=self._encode_value(self.item_encoder, item_value),
                        category_id=self._encode_value(
                            self.category_encoder, origin_row.get("category", "unknown")
                        ),
                        target=torch.tensor(
                            (target.fillna(0.0) / item_scale).to_numpy(dtype="float32"), dtype=torch.float32
                        ),
                        forecast_origin=pd.Timestamp(origin_row["date"]),
                        target_dates=pd.DatetimeIndex(target_rows["date"]),
                    )
                )
        return samples

    def _numeric_tensor(self, rows: pd.DataFrame, columns: Sequence[str]) -> Tensor:
        values = np.zeros((len(rows), len(columns)), dtype="float32")
        for position, column in enumerate(columns):
            numeric = (
                pd.to_numeric(rows[column], errors="coerce")
                if column in rows.columns
                else pd.Series(np.nan, index=rows.index, dtype="float64")
            )
            parameters = self.numeric_stats.get(column, {"mean": 0.0, "scale": 1.0})
            filled = numeric.fillna(float(parameters["mean"]))
            values[:, position] = (
                (filled.to_numpy(dtype="float64") - float(parameters["mean"]))
                / float(parameters["scale"] or 1.0)
            ).astype("float32")
        return torch.from_numpy(values)

    @staticmethod
    def _is_daily(dates: pd.Series) -> bool:
        expected = pd.Series(pd.date_range(dates.iloc[0], periods=len(dates), freq="D"))
        return dates.reset_index(drop=True).equals(expected)

    @staticmethod
    def _encode_value(mapping: Mapping[str, int], value: object) -> int:
        return int(mapping.get(str(value), 0))


def lstm_collate(samples: Iterable[LSTMWindowSample]) -> LSTMBatch:
    values = list(samples)
    if not values:
        raise ValueError("cannot collate an empty sample list")
    return LSTMBatch(
        history_numeric=torch.stack([sample.history_numeric for sample in values]),
        future_known=torch.stack([sample.future_known for sample in values]),
        item_id=torch.tensor([sample.item_id for sample in values], dtype=torch.long),
        category_id=torch.tensor([sample.category_id for sample in values], dtype=torch.long),
        target=torch.stack([sample.target for sample in values]),
        forecast_origins=tuple(sample.forecast_origin for sample in values),
        target_dates=tuple(sample.target_dates for sample in values),
    )


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: LSTMBatch, device: torch.device) -> LSTMBatch:
    return LSTMBatch(
        history_numeric=batch.history_numeric.to(device),
        future_known=batch.future_known.to(device),
        item_id=batch.item_id.to(device),
        category_id=batch.category_id.to(device),
        target=batch.target.to(device),
        forecast_origins=batch.forecast_origins,
        target_dates=batch.target_dates,
    )


class GlobalLSTM(nn.Module):
    """Global encoder with a direct seven-output non-negative head."""

    def __init__(
        self,
        history_features: int,
        future_features: int,
        item_count: int,
        category_count: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(item_count, 8)
        self.category_embedding = nn.Embedding(category_count, 4)
        self.encoder = nn.LSTM(
            history_features + 12,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 7 * future_features + 12, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7),
            nn.Softplus(),
        )

    def forward(self, batch: LSTMBatch | SimpleNamespace) -> Tensor:
        device = self.item_embedding.weight.device
        item_id = batch.item_id.to(device)
        category_id = batch.category_id.to(device)
        history_numeric = batch.history_numeric.to(device)
        future_known = batch.future_known.to(device)
        static = torch.cat(
            [self.item_embedding(item_id), self.category_embedding(category_id)], dim=-1
        )
        repeated = static.unsqueeze(1).expand(-1, history_numeric.size(1), -1)
        _, (hidden, _) = self.encoder(torch.cat([history_numeric, repeated], dim=-1))
        joined = torch.cat([hidden[-1], future_known.flatten(1), static], dim=-1)
        return self.head(joined)


class LSTMForecastModel:
    """Train and serve one global direct seven-day LSTM candidate."""

    def __init__(self, config: LSTMConfig) -> None:
        self.config = config
        self.model_id = config.name
        self.network: GlobalLSTM | None = None
        self.history_columns_: list[str] = []
        self.future_columns_: list[str] = []
        self.numeric_stats_: dict[str, dict[str, float]] = {}
        self.item_encoder_: dict[str, int] = {}
        self.category_encoder_: dict[str, int] = {}
        self.item_scales_: dict[str, float] = {}
        self.training_history_: list[dict[str, float | int]] = []
        self._fitted = False

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None:
        self._seed_everything(self.config.seed)
        prepared = LSTMWindowDataset._prepare_frame(history)
        self.history_columns_ = LSTMWindowDataset._infer_history_columns(prepared)
        self.future_columns_ = self._resolve_future_columns(prepared, known_covariates)
        self.numeric_stats_ = self._fit_numeric_stats(
            prepared, [*self.history_columns_, *self.future_columns_]
        )
        self.item_encoder_ = LSTMWindowDataset._build_encoder(prepared["item_id"])
        self.category_encoder_ = LSTMWindowDataset._build_encoder(prepared["category"])
        
        means = prepared.groupby("item_id", dropna=False)["quantity"].mean()
        self.item_scales_ = {str(item): float(val) + 1.0 for item, val in means.items()}
        
        dataset = LSTMWindowDataset(
            prepared,
            input_window=self.config.input_window,
            numeric_stats=self.numeric_stats_,
            horizon=HORIZON_DAYS,
            item_encoder=self.item_encoder_,
            category_encoder=self.category_encoder_,
            item_scales=self.item_scales_,
            history_columns=self.history_columns_,
            future_columns=self.future_columns_,
        )
        if not dataset:
            raise ValueError("history does not contain a complete LSTM training window")
        device = _get_device()
        self.network = GlobalLSTM(
            history_features=len(self.history_columns_),
            future_features=len(self.future_columns_),
            item_count=max(len(self.item_encoder_), 1),
            category_count=max(len(self.category_encoder_), 1),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(device)
        loader = self._loader(dataset)
        initial_loss = self._loss_over(loader)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()
        validation_start = max(1, int(len(dataset) * 0.8))
        train_indices = list(range(validation_start)) or list(range(len(dataset)))
        validation_indices = list(range(validation_start, len(dataset))) or train_indices
        train_loader = self._loader(Subset(dataset, train_indices), shuffle=False)
        validation_loader = self._loader(Subset(dataset, validation_indices), shuffle=False)
        self.training_history_ = [
            {"epoch": 0, "loss": initial_loss, "wape": self._wape_over(loader)}
        ]
        best_wape = float("inf")
        stale_epochs = 0
        for epoch in range(1, self.config.max_epochs + 1):
            self.network.train()
            losses: list[float] = []
            for batch in train_loader:
                batch_dev = _move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(batch_dev), batch_dev.target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip_val
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses)) if losses else initial_loss
            validation_wape = self._wape_over(validation_loader)
            self.training_history_.append(
                {"epoch": epoch, "loss": epoch_loss, "wape": validation_wape}
            )
            if epoch == 1 or epoch % 5 == 0 or epoch == self.config.max_epochs:
                print(f"    [LSTM: {self.model_id}] Epoch {epoch:3d}/{self.config.max_epochs} | Train Loss: {epoch_loss:.4f} | Val WAPE: {validation_wape:.4f}")
            if validation_wape + 1e-12 < best_wape:
                best_wape = validation_wape
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    print(f"    [LSTM: {self.model_id}] Early stopping at epoch {epoch} (best Val WAPE: {best_wape:.4f})")
                    break
        self._fitted = True

    def predict(self, history: pd.DataFrame, future_covariates: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.network is None:
            raise RuntimeError("LSTMForecastModel must be fit before predict")
        requested = self._validate_future(future_covariates)
        origin = requested["target_date"].min() - pd.Timedelta(1, unit="D")
        history_frame = LSTMWindowDataset._prepare_frame(history)
        rows: list[dict[str, object]] = []
        device = _get_device()
        self.network.to(device)
        self.network.eval()
        with torch.no_grad():
            for item_value, future_rows in requested.groupby("item_id", sort=True, dropna=False):
                item_history = history_frame.loc[
                    (history_frame["item_id"] == item_value) & (history_frame["date"] <= origin)
                ].sort_values("date")
                if len(item_history) < self.config.input_window:
                    raise ValueError(f"history is too short for item {item_value!r}")
                window = item_history.tail(self.config.input_window)
                category = str(window.iloc[-1].get("category", "unknown"))
                batch = LSTMBatch(
                    history_numeric=self._numeric_tensor(window, self.history_columns_),
                    future_known=self._numeric_tensor(future_rows, self.future_columns_),
                    item_id=torch.tensor([self._encoded_or_zero(self.item_encoder_, item_value)]),
                    category_id=torch.tensor(
                        [self._encoded_or_zero(self.category_encoder_, category)]
                    ),
                    target=torch.zeros((1, HORIZON_DAYS), dtype=torch.float32),
                    forecast_origins=(origin,),
                    target_dates=(pd.DatetimeIndex(future_rows["target_date"]),),
                )
                batch_dev = _move_batch_to_device(batch, device)
                values = self.network(batch_dev).squeeze(0).cpu().numpy()
                item_scale = self.item_scales_.get(str(item_value), 1.0)
                values = values * item_scale
                rows.extend(
                    {
                        "forecast_origin": origin,
                        "target_date": target_date,
                        "item_id": item_value,
                        "yhat": float(max(value, 0.0)),
                        "model_id": self.model_id,
                    }
                    for target_date, value in zip(future_rows["target_date"], values, strict=True)
                )
        return pd.DataFrame(rows).loc[
            :, ["forecast_origin", "target_date", "item_id", "yhat", "model_id"]
        ]

    def save(self, path: Path) -> None:
        if not self._fitted or self.network is None:
            raise RuntimeError("LSTMForecastModel must be fit before save")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "torch_version": torch.__version__,
            "config": self.config.model_dump(),
            "history_columns": self.history_columns_,
            "future_columns": self.future_columns_,
            "numeric_stats": self.numeric_stats_,
            "item_encoder": self.item_encoder_,
            "category_encoder": self.category_encoder_,
            "item_scales": self.item_scales_,
            "training_history": self.training_history_,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        torch.save(self.network.state_dict(), destination / "model.pt")

    @classmethod
    def load(cls, path: Path) -> LSTMForecastModel:
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported LSTM artifact schema version")
        if metadata.get("torch_version") != torch.__version__:
            raise ValueError("PyTorch artifact package version mismatch")
        model = cls(LSTMConfig.model_validate(metadata["config"]))
        model.history_columns_ = [str(value) for value in metadata["history_columns"]]
        model.future_columns_ = [str(value) for value in metadata["future_columns"]]
        model.numeric_stats_ = {
            str(column): {"mean": float(values["mean"]), "scale": float(values["scale"])}
            for column, values in metadata["numeric_stats"].items()
        }
        model.item_encoder_ = {
            str(value): int(code) for value, code in metadata["item_encoder"].items()
        }
        model.category_encoder_ = {
            str(value): int(code) for value, code in metadata["category_encoder"].items()
        }
        model.item_scales_ = {
            str(value): float(scale) for value, scale in metadata.get("item_scales", {}).items()
        }
        model.training_history_ = list(metadata.get("training_history", []))
        model.network = GlobalLSTM(
            history_features=len(model.history_columns_),
            future_features=len(model.future_columns_),
            item_count=max(len(model.item_encoder_), 1),
            category_count=max(len(model.category_encoder_), 1),
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_layers,
            dropout=model.config.dropout,
        )
        device = _get_device()
        model.network.load_state_dict(
            torch.load(source / "model.pt", map_location=device, weights_only=True)
        )
        model.network.to(device)
        model._fitted = True
        return model

    def _loader(self, dataset: Dataset[LSTMWindowSample], *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lstm_collate,
        )

    def _loss_over(self, loader: DataLoader) -> float:
        if self.network is None:
            raise RuntimeError("network is not initialized")
        device = _get_device()
        self.network.to(device)
        self.network.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            losses = [
                float(criterion(self.network(_move_batch_to_device(batch, device)), batch.target.to(device)).detach().cpu())
                for batch in loader
            ]
        return float(np.mean(losses))

    def _wape_over(self, loader: DataLoader) -> float:
        if self.network is None:
            raise RuntimeError("network is not initialized")
        device = _get_device()
        self.network.to(device)
        self.network.eval()
        numerator = denominator = 0.0
        with torch.no_grad():
            for batch in loader:
                batch_dev = _move_batch_to_device(batch, device)
                prediction = self.network(batch_dev).cpu().numpy()
                actual = batch_dev.target.cpu().numpy()
                numerator += float(np.abs(prediction - actual).sum())
                denominator += float(np.abs(actual).sum())
        return numerator / denominator if denominator else numerator

    def _numeric_tensor(self, rows: pd.DataFrame, columns: Sequence[str]) -> Tensor:
        dataset = object.__new__(LSTMWindowDataset)
        dataset.numeric_stats = self.numeric_stats_
        return LSTMWindowDataset._numeric_tensor(dataset, rows, columns).unsqueeze(0)

    def _resolve_future_columns(self, history: pd.DataFrame, known: pd.DataFrame) -> list[str]:
        supplied = [
            column
            for column in known.columns
            if column not in {"date", "target_date", "item_id", "category", "quantity"}
            and pd.api.types.is_numeric_dtype(known[column])
        ]
        return supplied or LSTMWindowDataset._infer_future_columns(history)

    @staticmethod
    def _fit_numeric_stats(
        frame: pd.DataFrame, columns: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for column in dict.fromkeys(columns):
            values = (
                pd.to_numeric(frame[column], errors="coerce").dropna()
                if column in frame
                else pd.Series(dtype="float64")
            )
            mean = float(values.mean()) if not values.empty else 0.0
            scale = float(values.std(ddof=0)) if not values.empty else 1.0
            stats[column] = {"mean": mean, "scale": scale or 1.0}
        return stats

    @staticmethod
    def _validate_future(future: pd.DataFrame) -> pd.DataFrame:
        missing = sorted({"target_date", "item_id"}.difference(future.columns))
        if missing:
            raise ValueError(f"future_covariates missing columns: {missing}")
        requested = future.copy()
        requested["target_date"] = pd.to_datetime(
            requested["target_date"], errors="raise"
        ).dt.normalize()
        if requested.duplicated(["item_id", "target_date"]).any():
            raise ValueError("future_covariates must have unique item and target-date keys")
        expected = pd.date_range(requested["target_date"].min(), periods=HORIZON_DAYS, freq="D")
        for _, rows in requested.groupby("item_id", sort=False, dropna=False):
            if set(rows["target_date"]) != set(expected):
                raise ValueError("each item must contain exact seven target dates")
        return requested.sort_values(["item_id", "target_date"]).reset_index(drop=True)

    @staticmethod
    def _encoded_or_zero(mapping: Mapping[str, int], value: object) -> int:
        return int(mapping.get(str(value), 0))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


__all__ = [
    "GlobalLSTM",
    "LSTMConfig",
    "LSTMForecastModel",
    "LSTMWindowDataset",
    "LSTMWindowSample",
    "lstm_collate",
]
