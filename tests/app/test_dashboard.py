import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "forecasts" / "2025-12-24"
    root.mkdir(parents=True)
    dates = pd.date_range("2025-12-25", periods=2)
    pd.DataFrame(
        {
            "target_date": dates,
            "item_id": ["coffee", "coffee"],
            "yhat": [10.0, 12.0],
            "lower": [8.0, 10.0],
            "upper": [13.0, 15.0],
            "model_id": ["seasonal_naive_7"] * 2,
            "fallback_used": [True, True],
            "fallback_reason": ["weather unavailable"] * 2,
        }
    ).to_csv(root / "item_forecasts.csv", index=False)
    pd.DataFrame(
        {
            "target_date": dates,
            "item_id": ["coffee", "coffee"],
            "forecast_revenue": [300000.0, 360000.0],
        }
    ).to_csv(root / "revenue_forecast.csv", index=False)
    pd.DataFrame(
        {
            "target_date": dates,
            "ingredient_id": ["beans", "beans"],
            "unit": ["kg", "kg"],
            "forecast_need": [0.2, 0.24],
        }
    ).to_csv(root / "ingredient_forecast.csv", index=False)
    pd.DataFrame(
        {
            "ingredient_id": ["beans"],
            "suggested_order": [1.0],
            "warning": ["expiry soon"],
        }
    ).to_csv(root / "order_proposals.csv", index=False)
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "provenance": "synthetic_vietnam_scenario",
                "dataset_name": "synthetic_vietnam_scenario",
                "forecast_origin": "2025-12-24",
                "horizon_days": 7,
                "fallback_used": True,
                "actual_model": "seasonal_naive_7",
                "fallback_reason": "weather unavailable",
                "warnings": ["weather unavailable"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FNB_FORECAST_OUTPUT_ROOT", str(tmp_path / "forecasts"))
    return root


def test_home_displays_synthetic_provenance_and_core_metrics(output_root: Path) -> None:
    app = AppTest.from_file(
        Path(__file__).parents[2] / "src/fnb_forecast/app/Home.py"
    ).run(timeout=30)

    assert not app.exception
    assert any("synthetic_vietnam_scenario" in str(element.value) for element in app.info)
    assert {metric.label for metric in app.metric} >= {
        "Doanh thu dự kiến",
        "Số ly dự kiến",
        "Cảnh báo kho",
    }


def test_all_pages_start_without_exception(output_root: Path) -> None:
    paths = [
        "src/fnb_forecast/app/Home.py",
        "src/fnb_forecast/app/pages/1_Item_Forecast.py",
        "src/fnb_forecast/app/pages/2_Ingredient_Orders.py",
        "src/fnb_forecast/app/pages/3_Model_Evaluation.py",
    ]
    for path in paths:
        app = AppTest.from_file(Path(__file__).parents[2] / path).run(timeout=30)
        assert not app.exception, path
