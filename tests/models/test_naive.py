import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fnb_forecast.models import SeasonalNaiveModel


def test_seasonal_naive_repeats_previous_week(
    make_model_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    history, future = make_model_frames(values=list(range(1, 15)), horizon=7)
    model = SeasonalNaiveModel(season_length=7)

    model.fit(history, future.iloc[0:0])
    result = model.predict(history, future)

    assert result["yhat"].tolist() == list(range(8, 15))
    assert result["model_id"].unique().tolist() == ["seasonal_naive_7"]


def test_seasonal_naive_uses_exact_date_lag_and_records_median_fallback(
    make_model_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    history, future = make_model_frames(values=list(range(1, 15)), horizon=7)
    history = history.loc[history["date"] != pd.Timestamp("2025-01-10")].copy()
    model = SeasonalNaiveModel(season_length=7)

    model.fit(history, future.iloc[0:0])
    result = model.predict(history, future)

    assert result["yhat"].tolist() == [8.0, 9.0, 7.0, 11.0, 12.0, 13.0, 14.0]
    assert model.metadata["fallback_reason"] == "missing_seasonal_lag"


def test_seasonal_naive_saves_versioned_metadata(
    tmp_path: Path,
    make_model_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    history, future = make_model_frames(values=list(range(1, 15)), horizon=7)
    model = SeasonalNaiveModel(season_length=7)
    model.fit(history, future.iloc[0:0])

    model.save(tmp_path)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["model_id"] == "seasonal_naive_7"
    assert metadata["season_length"] == 7
