import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from fnb_forecast.contracts import DataBundle
from fnb_forecast.evaluation.backtest import BacktestRunner
from fnb_forecast.evaluation.splits import ForecastFold
from fnb_forecast.features import FeatureBuilder
from fnb_forecast.models import (
    LightGBMConfig,
    LightGBMForecastModel,
    build_horizon_training_table,
)


def make_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    history_dates = pd.date_range("2025-01-01", periods=42, freq="D")
    all_dates = pd.date_range(history_dates.min(), periods=49, freq="D")
    items = ["coffee", "tea"]
    history = pd.MultiIndex.from_product(
        [items, history_dates], names=["item_id", "date"]
    ).to_frame(index=False)
    day = history.groupby("item_id").cumcount()
    history["quantity"] = day * 3 + history["item_id"].map({"coffee": 5, "tea": 11})
    history["category"] = history["item_id"].map(
        {"coffee": "hot_drink", "tea": "cold_drink"}
    )
    history["unit_price"] = history["item_id"].map({"coffee": 30_000.0, "tea": 25_000.0})
    history["promo_flag"] = pd.Series(day.mod(5).eq(0), dtype="boolean")
    history["stockout_flag"] = pd.Series(False, index=history.index, dtype="boolean")
    history["store_open"] = pd.Series(True, index=history.index, dtype="boolean")

    calendar = pd.DataFrame(
        {
            "date": all_dates,
            "weekday": all_dates.weekday,
            "is_weekend": pd.Series(all_dates.weekday >= 5, dtype="boolean"),
            "holiday_name": [None] * len(all_dates),
            "days_to_tet": range(60, 11, -1),
            "days_after_tet": range(-60, -11),
            "store_open": pd.Series(True, index=range(len(all_dates)), dtype="boolean"),
        }
    )
    origin = history_dates.max()
    target_dates = pd.date_range(origin + pd.Timedelta(1, unit="D"), periods=7)
    future = pd.MultiIndex.from_product(
        [items, target_dates], names=["item_id", "target_date"]
    ).to_frame(index=False)
    future["unit_price"] = future["item_id"].map({"coffee": 30_000.0, "tea": 25_000.0})
    future["promo_flag"] = pd.Series(False, index=future.index, dtype="boolean")
    weather = pd.DataFrame(
        {
            "issued_at": pd.Timestamp(origin, tz="Asia/Ho_Chi_Minh")
            - pd.Timedelta(1, unit="h"),
            "target_date": target_dates,
            "temperature": np.linspace(29.0, 32.0, 7),
            "rain": np.linspace(0.0, 6.0, 7),
            "humidity": np.linspace(65.0, 71.0, 7),
            "source": "previous_runs_fixture",
        }
    )
    return FeatureBuilder(calendar=calendar, weather=weather).build(history, future, origin)


def make_lightgbm_config(seed: int = 17) -> LightGBMConfig:
    return LightGBMConfig(
        name="lgb_test",
        n_estimators=20,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=8,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        seed=seed,
    )


def test_training_table_has_one_row_per_origin_item_horizon() -> None:
    feature_history, _ = make_feature_frames()

    table = build_horizon_training_table(feature_history, max_horizon=7)

    assert set(table["horizon"].unique()) == set(range(1, 8))
    assert (table["target_date"] - table["forecast_origin"]).dt.days.equals(
        table["horizon"]
    )
    assert table.groupby(["item_id", "forecast_origin", "horizon"]).size().eq(1).all()
    assert table["target_date"].le(feature_history["date"].max()).all()


def test_training_table_uses_origin_features_and_requires_all_targets() -> None:
    feature_history, _ = make_feature_frames()
    incomplete_origin = pd.Timestamp("2025-02-01")
    incomplete = feature_history.loc[
        ~(
            feature_history["item_id"].eq("tea")
            & feature_history["date"].eq(incomplete_origin + pd.Timedelta(7, unit="D"))
        )
    ].copy()

    table = build_horizon_training_table(incomplete, max_horizon=7)

    assert not (
        table["item_id"].eq("tea") & table["forecast_origin"].eq(incomplete_origin)
    ).any()
    sample = table.loc[
        table["item_id"].eq("coffee")
        & table["forecast_origin"].eq(pd.Timestamp("2025-01-29"))
        & table["horizon"].eq(7)
    ].squeeze()
    origin = feature_history.loc[
        feature_history["item_id"].eq("coffee")
        & feature_history["date"].eq(pd.Timestamp("2025-01-29"))
    ].squeeze()
    target = feature_history.loc[
        feature_history["item_id"].eq("coffee")
        & feature_history["date"].eq(pd.Timestamp("2025-02-05"))
    ].squeeze()
    assert sample["lag_1"] == origin["lag_1"]
    assert sample["lag_1"] != target["lag_1"]
    assert sample["target"] == target["quantity"]


def test_lightgbm_predictions_are_repeatable_nonnegative_and_complete() -> None:
    feature_history, future = make_feature_frames()
    first = LightGBMForecastModel(make_lightgbm_config(seed=17))
    first.fit(feature_history, future.iloc[0:0])
    p1 = first.predict(feature_history, future)
    second = LightGBMForecastModel(make_lightgbm_config(seed=17))
    second.fit(feature_history, future.iloc[0:0])
    p2 = second.predict(feature_history, future)

    assert p1.equals(p2)
    assert len(p1) == 14
    assert set(p1.columns) == {
        "forecast_origin",
        "target_date",
        "item_id",
        "yhat",
        "model_id",
    }
    assert not p1.duplicated(["forecast_origin", "target_date", "item_id"]).any()
    assert np.isfinite(p1["yhat"]).all()
    assert p1["yhat"].ge(0).all()
    assert first.estimator.get_params()["n_jobs"] == -1
    for _, item_rows in p1.groupby("item_id"):
        assert item_rows["target_date"].tolist() == list(
            pd.date_range("2025-02-12", periods=7)
        )


def test_config_is_strict_and_yaml_contains_the_exact_candidate_grid() -> None:
    with pytest.raises(ValidationError):
        LightGBMConfig.model_validate(
            {**make_lightgbm_config().model_dump(), "unexpected": True}
        )
    with pytest.raises(ValidationError):
        LightGBMConfig.model_validate(
            {**make_lightgbm_config().model_dump(), "n_jobs": -1}
        )

    payload = yaml.safe_load(Path("configs/models/lightgbm.yaml").read_text(encoding="utf-8"))
    candidates = [LightGBMConfig.model_validate(candidate) for candidate in payload["candidates"]]

    assert len(candidates) == 12
    assert len({candidate.name for candidate in candidates}) == 12
    assert {
        (candidate.num_leaves, candidate.max_depth, candidate.learning_rate)
        for candidate in candidates
    } == {
        (leaves, depth, rate)
        for leaves in [15, 31, 63]
        for depth in [-1, 8]
        for rate in [0.03, 0.05]
    }
    assert all(
        candidate.n_estimators == (800 if candidate.learning_rate == 0.03 else 500)
        and candidate.subsample == 0.9
        and candidate.subsample_freq == 1
        and candidate.colsample_bytree == 0.9
        and candidate.seed == 20260813
        for candidate in candidates
    )


def test_artifact_reload_reproduces_predictions_and_rejects_feature_mismatch(
    tmp_path: Path,
) -> None:
    feature_history, future = make_feature_frames()
    fitted = LightGBMForecastModel(make_lightgbm_config(seed=29))
    fitted.fit(feature_history, future.iloc[0:0])
    expected = fitted.predict(feature_history, future)

    fitted.save(tmp_path)
    restored = LightGBMForecastModel.load(tmp_path)

    assert restored.predict(feature_history, future).equals(expected)
    assert {path.name for path in tmp_path.iterdir()} == {"metadata.json", "model.joblib"}
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["package_version"]
    assert len(metadata["training_checksum"]) == 64
    assert metadata["config"] == make_lightgbm_config(seed=29).model_dump()
    assert metadata["feature_order"] == fitted.feature_order_
    assert metadata["category_encoders"] == fitted.category_encoders_
    assert "training_targets" not in metadata
    assert hasattr(joblib.load(tmp_path / "model.joblib"), "booster_")

    with pytest.raises(ValueError, match="missing required model features"):
        restored.predict(feature_history.drop(columns="lag_28"), future)


def test_lightgbm_backtest_keeps_fit_before_final_holdout_but_uses_as_of_features(
    baseline_backtest_case: tuple[DataBundle, list[ForecastFold]],
) -> None:
    bundle, folds = baseline_backtest_case
    fixed_train_test = replace(
        folds[1],
        name="test_01",
        role="test",
        train_end=folds[0].train_end,
    )

    result = BacktestRunner().run(
        lambda: LightGBMForecastModel(make_lightgbm_config()),
        bundle,
        [folds[0], fixed_train_test],
    )

    assert len(result) == 2 * 2 * 7
    assert result.loc[result["fold_name"].eq("test_01"), "yhat"].notna().all()
    assert result["yhat"].ge(0.0).all()
