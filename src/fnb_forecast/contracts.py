from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from .exceptions import DataValidationError

FORECAST_COLUMNS = {"forecast_origin", "target_date", "item_id", "yhat", "model_id"}


@dataclass(frozen=True)
class DataProvenance:
    name: str
    source: str
    is_synthetic: bool
    generated_at: datetime | None
    checksum: str | None


@dataclass
class DataBundle:
    daily_sales: pd.DataFrame
    item_master: pd.DataFrame
    calendar: pd.DataFrame
    weather: pd.DataFrame
    recipes: pd.DataFrame
    ingredient_master: pd.DataFrame
    inventory_lots: pd.DataFrame
    scheduled_receipts: pd.DataFrame
    provenance: DataProvenance


@dataclass(frozen=True)
class ForecastRequest:
    forecast_origin: pd.Timestamp
    horizon_days: int
    dataset_name: str

    def __post_init__(self) -> None:
        if self.horizon_days != 7:
            raise ValueError("horizon_days must be 7")


class ForecastModel(Protocol):
    model_id: str

    def fit(self, history: pd.DataFrame, known_covariates: pd.DataFrame) -> None: ...

    def predict(self, history: pd.DataFrame, future_covariates: pd.DataFrame) -> pd.DataFrame: ...

    def save(self, path: Path) -> None: ...


def assert_forecast_frame(frame: pd.DataFrame) -> None:
    missing = FORECAST_COLUMNS.difference(frame.columns)
    if missing:
        raise DataValidationError("forecasts", sorted(missing), [], "missing columns")
    if frame["yhat"].isna().any():
        raise DataValidationError(
            "forecasts",
            ["yhat"],
            frame.index[frame["yhat"].isna()].tolist(),
            "NaN forecast",
        )
