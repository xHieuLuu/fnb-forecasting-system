import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from fnb_forecast.models import (
    GlobalLSTM,
    LSTMConfig,
    LSTMForecastModel,
    LSTMWindowDataset,
    lstm_collate,
)


def make_lstm_frame(days: int = 72, item_count: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows: list[dict[str, object]] = []
    for item_number in range(item_count):
        for day_number, date in enumerate(dates):
            weekly = [3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0][day_number % 7]
            rows.append(
                {
                    "date": date,
                    "item_id": f"item-{item_number}",
                    "category": "drink" if item_number == 0 else "food",
                    "quantity": weekly + item_number * 2.0,
                    "unit_price": 20.0 + item_number,
                    "promo_flag": day_number % 5 == 0,
                    "temperature": 28.0 + (day_number % 4),
                    "store_open": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def lstm_frame() -> pd.DataFrame:
    return make_lstm_frame()


def make_lstm_batch(batch_size: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        history_numeric=torch.ones(batch_size, 28, 6),
        future_known=torch.ones(batch_size, 7, 4),
        item_id=torch.zeros(batch_size, dtype=torch.long),
        category_id=torch.zeros(batch_size, dtype=torch.long),
        target=torch.ones(batch_size, 7),
    )


def make_lstm_config(*, seed: int = 17, max_epochs: int = 20) -> LSTMConfig:
    return LSTMConfig(
        name="lstm_test",
        input_window=28,
        hidden_size=16,
        num_layers=2,
        dropout=0.0,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=max_epochs,
        early_stopping_patience=10,
        gradient_clip_val=1.0,
        seed=seed,
    )


def test_window_dataset_stops_targets_at_origin_plus_seven(lstm_frame: pd.DataFrame) -> None:
    dataset = LSTMWindowDataset(lstm_frame, input_window=28, horizon=7)
    sample = dataset[0]
    assert sample.history_numeric.shape[0] == 28
    assert sample.future_known.shape[0] == 7
    assert sample.target.shape == (7,)
    assert sample.target_dates.min() > sample.forecast_origin
    assert sample.target_dates.max() == sample.forecast_origin + pd.Timedelta(days=7)


def test_closed_store_origins_are_not_eligible(lstm_frame: pd.DataFrame) -> None:
    origin = pd.Timestamp("2025-01-28")
    mask = (lstm_frame["item_id"] == "item-0") & (lstm_frame["date"] == origin)
    lstm_frame.loc[mask, "store_open"] = False
    dataset = LSTMWindowDataset(lstm_frame, input_window=28, horizon=7)
    assert not any(
        sample.forecast_origin == origin and sample.item_id == dataset.item_encoder["item-0"]
        for sample in dataset.samples
    )


def test_global_lstm_outputs_nonnegative_seven_day_vector() -> None:
    model = GlobalLSTM(6, 4, 15, 4, 16, 1, 0.0)
    output = model(make_lstm_batch(batch_size=3))
    assert output.shape == (3, 7)
    assert torch.all(output >= 0)


def test_lstm_config_yaml_contains_exact_candidate_grid() -> None:
    payload = yaml.safe_load(Path("configs/models/lstm.yaml").read_text(encoding="utf-8"))
    candidates = [LSTMConfig.model_validate(candidate) for candidate in payload["candidates"]]
    assert len(candidates) == 12
    assert len({candidate.name for candidate in candidates}) == 12
    assert {
        (candidate.input_window, candidate.hidden_size, candidate.dropout)
        for candidate in candidates
    } == {
        (window, hidden, dropout)
        for window in [28, 56]
        for hidden in [32, 64, 128]
        for dropout in [0.1, 0.3]
    }


def test_lstm_learns_tiny_batch_and_reloaded_artifact_is_equivalent(
    lstm_frame: pd.DataFrame, tmp_path: Path
) -> None:
    first = LSTMForecastModel(make_lstm_config(seed=19, max_epochs=20))
    first.fit(lstm_frame, lstm_frame.iloc[0:0])
    assert first.training_history_[-1]["loss"] < first.training_history_[0]["loss"]
    future = _future_for(lstm_frame)
    assert first.predict(lstm_frame, future)["yhat"].ge(0).all()
    first.save(tmp_path)
    restored = LSTMForecastModel.load(tmp_path)
    np.testing.assert_allclose(
        restored.predict(lstm_frame, future)["yhat"],
        first.predict(lstm_frame, future)["yhat"],
        atol=1e-5,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"] == make_lstm_config(seed=19, max_epochs=20).model_dump()
    assert (tmp_path / "model.pt").exists()


def test_lstm_same_seed_is_deterministic(lstm_frame: pd.DataFrame) -> None:
    first = LSTMForecastModel(make_lstm_config(seed=23, max_epochs=3))
    second = LSTMForecastModel(make_lstm_config(seed=23, max_epochs=3))
    first.fit(lstm_frame, lstm_frame.iloc[0:0])
    second.fit(lstm_frame, lstm_frame.iloc[0:0])
    np.testing.assert_allclose(
        first.predict(lstm_frame, _future_for(lstm_frame))["yhat"],
        second.predict(lstm_frame, _future_for(lstm_frame))["yhat"],
        atol=1e-5,
    )


def test_lstm_collate_preserves_tensor_shapes(lstm_frame: pd.DataFrame) -> None:
    dataset = LSTMWindowDataset(lstm_frame, input_window=28, horizon=7)
    batch = lstm_collate([dataset[0], dataset[1]])
    assert batch.history_numeric.shape == (2, 28, len(dataset.history_columns))
    assert batch.future_known.shape == (2, 7, len(dataset.future_columns))
    assert batch.target.shape == (2, 7)


def _future_for(frame: pd.DataFrame) -> pd.DataFrame:
    last_date = pd.to_datetime(frame["date"]).max()
    target_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    future = (
        frame.loc[:, ["item_id"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"target_date": target_dates}), how="cross")
    )
    future["unit_price"] = future["item_id"].map(frame.groupby("item_id")["unit_price"].first())
    future["promo_flag"] = False
    future["temperature"] = 29.0
    future["store_open"] = True
    return future
