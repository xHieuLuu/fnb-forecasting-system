from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fnb_forecast.models.tft import (
    TFTConfig,
    TFTForecastModel,
    TFTInterpretation,
    build_tft_dataset,
)


@dataclass(frozen=True)
class TFTFixture:
    history: pd.DataFrame
    known_covariates: pd.DataFrame
    future_covariates: pd.DataFrame


@pytest.fixture
def tft_frame() -> TFTFixture:
    dates = pd.date_range("2025-01-01", periods=64, freq="D")
    rows: list[dict[str, object]] = []
    for item_number, item_id in enumerate(["coffee", "tea"]):
        for day_number, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "item_id": item_id,
                    "category": "beverage",
                    "quantity": float(8 + item_number + day_number % 7),
                    "stockout_flag": 0.0,
                    "weekday": float(date.weekday()),
                    "month": float(date.month),
                    "promo_flag": float(day_number % 5 == 0),
                    "unit_price": float(30_000 + item_number * 1_000),
                    "temperature": float(27 + day_number % 4),
                    "rain": float(day_number % 3),
                    "humidity": float(68 + day_number % 5),
                }
            )
    history = pd.DataFrame(rows)
    target_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7)
    future = pd.MultiIndex.from_product(
        [["coffee", "tea"], target_dates], names=["item_id", "target_date"]
    ).to_frame(index=False)
    future["category"] = "beverage"
    future["weekday"] = future["target_date"].dt.weekday.astype(float)
    future["month"] = future["target_date"].dt.month.astype(float)
    future["promo_flag"] = 0.0
    future["unit_price"] = future["item_id"].map({"coffee": 30_000.0, "tea": 31_000.0})
    future["temperature"] = 29.0
    future["rain"] = 1.0
    future["humidity"] = 70.0
    return TFTFixture(history, history.copy(), future)


def make_tft_config() -> TFTConfig:
    return TFTConfig(
        name="tft_test",
        input_window=28,
        hidden_size=4,
        attention_head_size=1,
        hidden_continuous_size=2,
        dropout=0.1,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=1,
        early_stopping_patience=10,
        gradient_clip_val=1.0,
        limit_train_batches=2,
        seed=17,
    )


def test_tft_dataset_keeps_target_out_of_known_future(tft_frame: TFTFixture) -> None:
    dataset = build_tft_dataset(tft_frame.history, make_tft_config(), training=True)

    assert "quantity" in dataset.time_varying_unknown_reals
    assert "quantity" not in dataset.time_varying_known_reals
    assert {
        "weekday",
        "month",
        "promo_flag",
        "unit_price",
        "temperature",
        "rain",
        "humidity",
    }.issubset(dataset.time_varying_known_reals)
    assert dataset.max_encoder_length == 28
    assert dataset.max_prediction_length == 7


def test_tft_cpu_smoke_returns_seven_days_per_item(tft_frame: TFTFixture) -> None:
    adapter = TFTForecastModel(make_tft_config())
    adapter.fit(tft_frame.history, tft_frame.known_covariates)

    result = adapter.predict(tft_frame.history, tft_frame.future_covariates)

    assert result.groupby("item_id").size().eq(7).all()
    assert result["yhat"].ge(0).all()
    assert set(result.columns) == {
        "forecast_origin",
        "target_date",
        "item_id",
        "yhat",
        "model_id",
    }


def test_tft_interpretation_disclaims_causality(
    tft_frame: TFTFixture,
) -> None:
    adapter = TFTForecastModel(make_tft_config())
    adapter.fit(tft_frame.history, tft_frame.known_covariates)

    interpretation = adapter.interpret(tft_frame.history, tft_frame.future_covariates)

    assert isinstance(interpretation, TFTInterpretation)
    assert interpretation.metadata["causal_interpretation"] is False
    assert "cause" not in " ".join(interpretation.metadata.keys()).lower()
    assert set(interpretation.variable_importance.columns) >= {"variable", "importance"}


def test_tft_yaml_contains_exact_candidate_grid() -> None:
    payload = yaml.safe_load(Path("configs/models/tft.yaml").read_text(encoding="utf-8"))
    candidates = [TFTConfig.model_validate(candidate) for candidate in payload["candidates"]]

    assert len(candidates) == 12
    assert len({candidate.name for candidate in candidates}) == 12
    assert {
        (candidate.input_window, candidate.hidden_size, candidate.attention_head_size)
        for candidate in candidates
    } == {
        (window, hidden, heads)
        for window in [28, 56]
        for hidden in [16, 32, 64]
        for heads in [1, 4]
    }
    assert all(
        candidate.hidden_continuous_size == min(candidate.hidden_size // 2, 16)
        for candidate in candidates
    )


def test_tft_artifact_reload_reproduces_cpu_forecasts(
    tft_frame: TFTFixture, tmp_path: Path
) -> None:
    adapter = TFTForecastModel(make_tft_config())
    adapter.fit(tft_frame.history, tft_frame.known_covariates)
    expected = adapter.predict(tft_frame.history, tft_frame.future_covariates)
    adapter.save(tmp_path)

    restored = TFTForecastModel.load(tmp_path)
    actual = restored.predict(tft_frame.history, tft_frame.future_covariates)

    np.testing.assert_allclose(actual["yhat"], expected["yhat"], atol=1e-5)
    assert (tmp_path / "model.ckpt").exists()
    assert (tmp_path / "metadata.json").exists()
