# F&B Demand and Revenue Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng prototype tái lập có thể dự báo trực tiếp nhu cầu 15 món trong 7 ngày, quy đổi sang doanh thu/nguyên liệu, mô phỏng tồn kho và trình bày kết quả bằng Streamlit.

**Architecture:** Pipeline offline chịu trách nhiệm kiểm tra dữ liệu, tạo feature theo cutoff, backtesting, chọn model riêng cho M5 và kịch bản Việt Nam, rồi lưu artifact có metadata. Dịch vụ inference chỉ tải artifact đã chọn, tạo dự báo và gọi các hàm thuần cho BOM, doanh thu, tồn kho; Streamlit chỉ hiển thị/kích hoạt các dịch vụ này và không chứa logic mô hình.

**Tech Stack:** Python 3.11/3.12, pandas, NumPy, Pydantic, PyYAML, PyTorch, PyTorch Forecasting 1.x, Lightning, LightGBM, StatsForecast, scikit-learn, Streamlit, Plotly, httpx, python-holidays, Typer, pytest, Ruff, mypy.

## Global Constraints

- Hỗ trợ Python `>=3.11,<3.13` 64-bit trên Windows và Linux; mọi model phải chạy được trên CPU, GPU chỉ là tăng tốc tùy chọn.
- Khóa dependency trong `pyproject.toml`: `torch>=2.6,<3`, `pytorch-forecasting>=1.8,<2`, `lightgbm>=4.7,<5`, `statsforecast>=2,<3`, `streamlit>=1.60,<2`.
- Chân trời dự báo cố định 7 ngày; tần suất dữ liệu là ngày; input window DL chỉ gồm 28 hoặc 56 ngày.
- Kịch bản Việt Nam cố định 15 món, 15 nguyên liệu và 730 ngày từ `2024-01-02` đến `2025-12-31`.
- Tập test là 28 ngày cuối; validation gồm 8 cutoff expanding-window cách nhau 7 ngày; không fit scaler, encoder hoặc selector trên test.
- Mỗi bảng hoặc artifact phải chứa provenance; mọi màn hình dùng dữ liệu tổng hợp phải hiển thị `synthetic_vietnam_scenario`.
- Weather feature ở tương lai phải có `issued_at <= forecast_origin`; không thay bằng thời tiết quan sát thực tế.
- Dashboard chạy cục bộ, không đăng nhập, phân quyền, thanh toán hoặc tích hợp POS trực tiếp.
- Mọi thay đổi dùng TDD: test thất bại trước, implementation tối thiểu sau, test đạt rồi mới commit.
- Không commit dữ liệu M5, cache thời tiết, model weights hoặc output thí nghiệm; chỉ commit script, checksum manifest mẫu và tài liệu.

## File and Interface Map

```text
pyproject.toml                         Package, dependencies, CLI entry point
README.md                              Setup and command sequence
.gitignore                             Runtime/data/artifact exclusions
configs/base.yaml                      Project-wide constants
configs/models/*.yaml                  Twelve bounded candidates per learned model
data/README.md                         M5 acquisition, license, checksum workflow
src/fnb_forecast/config.py             Pydantic project configuration
src/fnb_forecast/contracts.py          Shared dataclasses, protocols, schemas
src/fnb_forecast/exceptions.py         Typed domain errors
src/fnb_forecast/cli.py                Typer commands
src/fnb_forecast/data/io.py            CSV bundle loading and persistence
src/fnb_forecast/data/validation.py     Table validation and daily completion
src/fnb_forecast/data/calendar.py       Vietnam calendar features
src/fnb_forecast/data/weather.py        Open-Meteo retrieval/cache/fallback signal
src/fnb_forecast/data/synthetic.py      Deterministic Vietnam scenario generator
src/fnb_forecast/data/m5.py             M5 normalization and leakage-safe subset
src/fnb_forecast/features/builder.py    As-of lag/rolling/known-covariate features
src/fnb_forecast/evaluation/splits.py   Validation and final-test forecast origins
src/fnb_forecast/evaluation/metrics.py  WAPE, MAE, RMSSE, bias and bootstrap CI
src/fnb_forecast/evaluation/backtest.py Shared rolling-origin runner
src/fnb_forecast/models/base.py          ForecastModel abstract adapter
src/fnb_forecast/models/naive.py         Seasonal Naive adapter
src/fnb_forecast/models/ets.py           StatsForecast AutoETS adapter
src/fnb_forecast/models/lightgbm.py      Global horizon-index LightGBM adapter
src/fnb_forecast/models/lstm.py          Global direct seven-output LSTM adapter
src/fnb_forecast/models/tft.py           PyTorch Forecasting TFT adapter
src/fnb_forecast/models/registry.py      Artifact save/load and compatibility checks
src/fnb_forecast/evaluation/experiment.py Grid execution, ablation and selection
src/fnb_forecast/evaluation/uncertainty.py Residual intervals and safety-stock errors
src/fnb_forecast/planning/revenue.py     Revenue and BOM conversion
src/fnb_forecast/planning/inventory.py   FEFO, safety stock and order proposals
src/fnb_forecast/planning/simulation.py  Forecast-vs-moving-average policy simulation
src/fnb_forecast/services/forecast.py    Inference orchestration and explicit fallbacks
src/fnb_forecast/app/Home.py             Streamlit overview
src/fnb_forecast/app/pages/*.py          Item, inventory and model pages
tests/                                  Mirrored unit/integration/smoke tests
```

Shared contracts, introduced in Task 1 and reused verbatim:

```python
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
```

Every `predict()` returns exactly these columns:

```text
forecast_origin, target_date, item_id, yhat, model_id
```

`lower` and `upper` are added only by `ResidualIntervalCalibrator` after model prediction.

Test snippets use small local factories such as `valid_bundle`, `make_model_frames`, and `ingredient_master`. Each named factory is created at the top of the test file in the same step where it first appears; it must return only the exact columns consumed by the production signature in that task. Factories use fixed dates and seeds, never call the network, and are not imported from production modules. Shared pytest fixtures `project_config`, `observed_weather`, and `valid_bundle` live in `tests/conftest.py`, created in Task 1 and extended when Tasks 2–4 introduce their final schemas.

---

### Task 1: Project Foundation, Configuration, and Shared Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Modify: `.gitignore`
- Create: `configs/base.yaml`
- Create: `src/fnb_forecast/__init__.py`
- Create: `src/fnb_forecast/config.py`
- Create: `src/fnb_forecast/contracts.py`
- Create: `src/fnb_forecast/exceptions.py`
- Create: `src/fnb_forecast/cli.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Consumes: Không có; đây là nền móng của repository.
- Produces: `ProjectConfig`, `load_project_config(path: Path) -> ProjectConfig`, `DataBundle`, `DataProvenance`, `ForecastRequest`, `ForecastModel`, `DataValidationError`, `WeatherUnavailableError`, `ArtifactCompatibilityError`, `ModelInferenceError`.

- [ ] **Step 1: Write configuration and contract tests**

```python
# tests/test_config.py
from pathlib import Path
import pytest
from pydantic import ValidationError
from fnb_forecast.config import load_project_config

def test_base_config_locks_horizon_and_scenario_dates(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "forecast:\n  horizon_days: 7\n  input_windows: [28, 56]\n"
        "scenario:\n  start_date: 2024-01-02\n  end_date: 2025-12-31\n"
        "  seed: 20260813\n  item_count: 15\n  ingredient_count: 15\n",
        encoding="utf-8",
    )
    config = load_project_config(path)
    assert config.forecast.horizon_days == 7
    assert (config.scenario.end_date - config.scenario.start_date).days + 1 == 730

def test_horizon_other_than_seven_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "forecast:\n  horizon_days: 14\n  input_windows: [28, 56]\n"
        "scenario:\n  start_date: 2024-01-02\n  end_date: 2025-12-31\n"
        "  seed: 1\n  item_count: 15\n  ingredient_count: 15\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_project_config(path)
```

```python
# tests/test_contracts.py
import pandas as pd
from fnb_forecast.contracts import assert_forecast_frame
from fnb_forecast.exceptions import DataValidationError

def test_forecast_contract_rejects_missing_item_id() -> None:
    frame = pd.DataFrame({"forecast_origin": ["2025-01-01"], "target_date": ["2025-01-02"], "yhat": [1.0], "model_id": ["naive"]})
    try:
        assert_forecast_frame(frame)
    except DataValidationError as exc:
        assert "item_id" in str(exc)
    else:
        raise AssertionError("missing item_id must fail")
```

- [ ] **Step 2: Run tests and verify the package is missing**

Run: `python -m pytest tests/test_config.py tests/test_contracts.py -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'fnb_forecast'`.

- [ ] **Step 3: Add packaging, dependencies, configuration, and contracts**

Use this exact Pydantic shape in `src/fnb_forecast/config.py`:

```python
from datetime import date
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ForecastConfig(StrictModel):
    horizon_days: Literal[7]
    input_windows: tuple[Literal[28, 56], ...]

class ScenarioConfig(StrictModel):
    start_date: date
    end_date: date
    seed: int
    item_count: Literal[15]
    ingredient_count: Literal[15]

    @model_validator(mode="after")
    def require_730_days(self):
        if (self.end_date - self.start_date).days + 1 != 730:
            raise ValueError("scenario must contain exactly 730 days")
        return self

class EvaluationConfig(StrictModel):
    validation_folds: Literal[8] = 8
    test_days: Literal[28] = 28

class LocationConfig(StrictModel):
    latitude: float = 10.8231
    longitude: float = 106.6297
    timezone: Literal["Asia/Ho_Chi_Minh"] = "Asia/Ho_Chi_Minh"

class PathsConfig(StrictModel):
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    cache_root: Path = Path(".cache")

class ProjectConfig(StrictModel):
    forecast: ForecastConfig
    scenario: ScenarioConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

def load_project_config(path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

In `contracts.py`, add the shared dataclasses/protocol from the interface map and:

```python
FORECAST_COLUMNS = {"forecast_origin", "target_date", "item_id", "yhat", "model_id"}

def assert_forecast_frame(frame: pd.DataFrame) -> None:
    missing = FORECAST_COLUMNS.difference(frame.columns)
    if missing:
        raise DataValidationError("forecasts", sorted(missing), [], "missing columns")
    if frame["yhat"].isna().any():
        raise DataValidationError("forecasts", ["yhat"], frame.index[frame["yhat"].isna()].tolist(), "NaN forecast")
```

Define `DataValidationError` with `table`, `columns`, `rows`, and `reason` attributes so UI code never parses error strings. Also define typed `WeatherUnavailableError`, `ArtifactCompatibilityError`, and `ModelInferenceError`, all inheriting from a common `ForecastingDomainError`; adapters must wrap third-party exceptions in one of these types while preserving the original exception as `__cause__`. Add `pyproject.toml` with a `src` layout, the dependency ranges in Global Constraints, `fnb-forecast = "fnb_forecast.cli:app"`, and dev dependencies `pytest`, `pytest-cov`, `ruff`, and `mypy`. Add `data/raw/`, `data/processed/`, `artifacts/`, `.cache/`, `.venv/`, and `__pycache__/` to `.gitignore`.

Start `tests/conftest.py` with a fixed configuration fixture:

```python
@pytest.fixture
def project_config() -> ProjectConfig:
    return ProjectConfig.model_validate({
        "forecast": {"horizon_days": 7, "input_windows": [28, 56]},
        "scenario": {"start_date": "2024-01-02", "end_date": "2025-12-31", "seed": 20260813, "item_count": 15, "ingredient_count": 15},
    })
```

- [ ] **Step 4: Install the editable package and run foundation checks**

Run: `python -m pip install -e ".[dev]"`

Run: `python -m pytest tests/test_config.py tests/test_contracts.py -v`

Run: `python -m ruff check src tests`

Expected: installation succeeds; both tests PASS; Ruff reports no errors.

- [ ] **Step 5: Commit the foundation**

```powershell
git add pyproject.toml README.md .gitignore configs/base.yaml src/fnb_forecast tests/conftest.py tests/test_config.py tests/test_contracts.py
git commit -m "build: initialize forecasting project contracts"
```

### Task 2: CSV Bundle Validation and Daily Normalization

**Files:**
- Create: `src/fnb_forecast/data/__init__.py`
- Create: `src/fnb_forecast/data/io.py`
- Create: `src/fnb_forecast/data/validation.py`
- Modify: `tests/conftest.py`
- Test: `tests/data/test_validation.py`
- Test: `tests/data/test_io.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `DataBundle`, `DataProvenance`, `DataValidationError` from Task 1.
- Produces: `validate_bundle(bundle: DataBundle) -> DataBundle`, `complete_daily_sales(sales: pd.DataFrame, items: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame`, `load_csv_bundle(root: Path, provenance: DataProvenance) -> DataBundle`, `save_bundle(bundle: DataBundle, root: Path) -> None`.

- [ ] **Step 1: Write validation tests for negative values and missing days**

```python
# tests/data/test_validation.py
import pandas as pd
import pytest
from fnb_forecast.data.validation import complete_daily_sales, validate_daily_sales
from fnb_forecast.exceptions import DataValidationError

def test_negative_quantity_reports_rows_and_column() -> None:
    sales = pd.DataFrame({"date": ["2025-01-01"], "item_id": ["coffee"], "quantity": [-1], "unit_price": [30000], "promo_flag": [False], "stockout_flag": [False], "store_open": [True]})
    with pytest.raises(DataValidationError) as caught:
        validate_daily_sales(sales)
    assert caught.value.columns == ["quantity"]
    assert caught.value.rows == [0]

def test_open_missing_day_becomes_zero_but_closed_day_stays_closed() -> None:
    sales = pd.DataFrame({"date": ["2025-01-01"], "item_id": ["coffee"], "quantity": [4], "unit_price": [30000], "promo_flag": [False], "stockout_flag": [False], "store_open": [True]})
    items = pd.DataFrame({"item_id": ["coffee"]})
    calendar = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]), "store_open": [True, True, False]})
    result = complete_daily_sales(sales, items, calendar)
    assert result.loc[result["date"].eq("2025-01-02"), "quantity"].item() == 0
    assert pd.isna(result.loc[result["date"].eq("2025-01-03"), "quantity"].item())
```

- [ ] **Step 2: Run the focused validation tests**

Run: `python -m pytest tests/data/test_validation.py -v`

Expected: FAIL because `fnb_forecast.data.validation` does not exist.

- [ ] **Step 3: Implement strict schemas and completion rules**

In `validation.py`, define required columns for all eight tables. Normalize dates with `pd.to_datetime(...).dt.normalize()`, booleans with nullable Boolean dtype, and reject duplicated `(date, item_id)` rows. Implement missing-day completion with a cartesian product of active items and calendar days. Assign `quantity=0` only where `store_open=True`; leave it `NaN` on closed days so models can exclude those rows.

```python
def validate_daily_sales(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "item_id", "quantity", "unit_price", "promo_flag", "stockout_flag", "store_open"}
    _require_columns("daily_sales", frame, required)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    for column in ("quantity", "unit_price"):
        bad = out.index[out[column].lt(0) | out[column].isna()].tolist()
        if bad:
            raise DataValidationError("daily_sales", [column], bad, "must be non-negative and non-null")
    duplicates = out.index[out.duplicated(["date", "item_id"], keep=False)].tolist()
    if duplicates:
        raise DataValidationError("daily_sales", ["date", "item_id"], duplicates, "duplicate key")
    return out.sort_values(["item_id", "date"]).reset_index(drop=True)
```

In `io.py`, map fixed filenames to `DataBundle` fields, require UTF-8 CSV, validate after loading, and write a `provenance.json` beside saved tables. Never infer synthetic provenance from a directory name.

Extend `tests/conftest.py` with `valid_bundle`: one item, one ingredient, three open days, one BOM row, one unexpired inventory lot, no scheduled receipts, and all eight tables containing their required columns. Reuse this fixture for every CSV round-trip test.

- [ ] **Step 4: Add round-trip tests and run the data test group**

```python
# tests/data/test_io.py
def test_bundle_round_trip_preserves_provenance(valid_bundle, tmp_path):
    save_bundle(valid_bundle, tmp_path)
    loaded = load_csv_bundle(tmp_path, valid_bundle.provenance)
    assert loaded.provenance == valid_bundle.provenance
    assert loaded.daily_sales.equals(valid_bundle.daily_sales)
```

Run: `python -m pytest tests/data -v`

Expected: all data validation and I/O tests PASS.

- [ ] **Step 5: Commit validated CSV ingestion**

```powershell
git add src/fnb_forecast/data tests/conftest.py tests/data README.md
git commit -m "feat: validate and load daily F&B data bundles"
```

### Task 3: Vietnam Calendar and Leakage-Safe Weather Retrieval

**Files:**
- Create: `src/fnb_forecast/data/calendar.py`
- Create: `src/fnb_forecast/data/weather.py`
- Modify: `tests/conftest.py`
- Test: `tests/data/test_calendar.py`
- Test: `tests/data/test_weather.py`
- Modify: `configs/base.yaml`

**Interfaces:**
- Consumes: normalized dates and `DataValidationError`.
- Produces: `build_vietnam_calendar(start: date, end: date) -> pd.DataFrame`, `OpenMeteoWeatherClient.fetch_historical_observations(start: date, end: date) -> pd.DataFrame`, `fetch_previous_runs_forecast(origin: pd.Timestamp, target_dates: pd.DatetimeIndex) -> pd.DataFrame`, `fetch_live_forecast(origin, target_dates)`, and `WeatherUnavailableError`.

- [ ] **Step 1: Write calendar and issued-at leakage tests**

```python
# tests/data/test_calendar.py
from datetime import date
from fnb_forecast.data.calendar import build_vietnam_calendar

def test_vietnam_calendar_marks_tet_and_distance_features() -> None:
    frame = build_vietnam_calendar(date(2025, 1, 27), date(2025, 2, 2))
    tet = frame.loc[frame["date"].eq("2025-01-29")].iloc[0]
    assert tet["is_holiday"]
    assert tet["days_to_tet"] == 0
    assert frame["days_to_tet"].tolist() == [2, 1, 0, -1, -2, -3, -4]
```

```python
# tests/data/test_weather.py
import pandas as pd
import pytest
from fnb_forecast.data.weather import assert_weather_known_at_origin
from fnb_forecast.exceptions import WeatherUnavailableError

def test_future_issued_weather_is_rejected() -> None:
    frame = pd.DataFrame({"issued_at": ["2025-01-03"], "target_date": ["2025-01-04"], "temperature": [31.0], "rain": [4.0], "humidity": [80.0], "source": ["historical_forecast"]})
    with pytest.raises(WeatherUnavailableError):
        assert_weather_known_at_origin(frame, pd.Timestamp("2025-01-02"))
```

- [ ] **Step 2: Run the tests and verify missing modules**

Run: `python -m pytest tests/data/test_calendar.py tests/data/test_weather.py -v`

Expected: FAIL because the functions do not exist.

- [ ] **Step 3: Implement calendar, HTTP client, and disk cache**

Use `holidays.country_holidays("VN", years=...)`, with Tết day defined as the first holiday label in each year whose Unicode-normalized lowercase value contains `tet`, `lunar new year`, or `vietnamese new year`; fail with a descriptive error if a requested year has no matching entry. The returned calendar columns are `date`, `weekday`, `week_of_year`, `month`, `is_weekend`, `is_holiday`, `holiday_name`, `days_to_tet`, and `days_after_tet`.

Use `httpx.Client` dependency injection so tests supply `MockTransport`. Cache each response under `.cache/weather/<sha256(request-url)>.json`. Historical observations use `archive-api.open-meteo.com`, fixed-lead forecasts use `previous-runs-api.open-meteo.com`, and live forecasts use `api.open-meteo.com`. For target horizon `h` from 1 to 7, request the matching variables suffixed `_previous_day<h>` and record `issued_at = target_date - h days`; never mix lead times. Normalize hourly data to daily mean temperature/humidity and daily rain sum. Historical observations are allowed only in the synthetic generator target, never in a model future-covariate frame. Run this guard before returning forecast features:

```python
def assert_weather_known_at_origin(frame: pd.DataFrame, origin: pd.Timestamp) -> None:
    issued = pd.to_datetime(frame["issued_at"], utc=True)
    origin = pd.Timestamp(origin)
    origin_utc = origin.tz_localize("Asia/Ho_Chi_Minh").tz_convert("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    if issued.gt(origin_utc).any():
        raise WeatherUnavailableError("weather issuance is later than forecast origin")
```

On HTTP error, return a valid cache entry if present; otherwise raise `WeatherUnavailableError`. Do not synthesize weather or fetch observations in this method.

Extend `tests/conftest.py` with `observed_weather`: a deterministic daily frame over the configured scenario dates containing `date`, `temperature`, `rain`, `humidity`, and `source="historical_observation_fixture"`.

- [ ] **Step 4: Test cache fallback with an HTTP mock**

Add a test that performs one successful mocked previous-runs request, changes the transport to return HTTP 503, and asserts the second call returns the cached frame with `source="previous_runs_cache"`.

Run: `python -m pytest tests/data/test_calendar.py tests/data/test_weather.py -v`

Expected: all tests PASS without internet access.

- [ ] **Step 5: Commit external covariates**

```powershell
git add src/fnb_forecast/data/calendar.py src/fnb_forecast/data/weather.py tests/conftest.py tests/data configs/base.yaml
git commit -m "feat: add Vietnam calendar and weather covariates"
```

### Task 4: Deterministic Vietnam F&B Scenario Generator

**Files:**
- Create: `configs/synthetic.yaml`
- Create: `src/fnb_forecast/data/synthetic.py`
- Create: `tests/data/test_synthetic.py`
- Modify: `src/fnb_forecast/cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ProjectConfig`, `SyntheticScenarioDefinition`, `DataBundle`, `DataProvenance`, `build_vietnam_calendar`, observed daily weather frame.
- Produces: `load_synthetic_definition(path: Path) -> SyntheticScenarioDefinition`, `generate_scenario(project: ProjectConfig, definition: SyntheticScenarioDefinition, observed_weather: pd.DataFrame) -> DataBundle`, and CLI `fnb-forecast generate-synthetic --project-config configs/base.yaml --scenario-config configs/synthetic.yaml --output data/processed/synthetic`.

- [ ] **Step 1: Write deterministic shape and provenance tests**

```python
# tests/data/test_synthetic.py
import pandas as pd

def test_scenario_has_exact_scope_and_is_deterministic(project_config, scenario_definition, observed_weather):
    first = generate_scenario(project_config, scenario_definition, observed_weather)
    second = generate_scenario(project_config, scenario_definition, observed_weather)
    assert first.item_master["item_id"].nunique() == 15
    assert first.ingredient_master["ingredient_id"].nunique() == 15
    assert len(first.daily_sales) == 15 * 730
    assert first.daily_sales.equals(second.daily_sales)
    assert first.provenance.name == "synthetic_vietnam_scenario"
    assert first.provenance.is_synthetic is True

def test_synthetic_sales_are_nonnegative_and_closed_days_are_nan(project_config, scenario_definition, observed_weather):
    bundle = generate_scenario(project_config, scenario_definition, observed_weather)
    open_rows = bundle.daily_sales["store_open"]
    assert bundle.daily_sales.loc[open_rows, "quantity"].ge(0).all()
    assert bundle.daily_sales.loc[~open_rows, "quantity"].isna().all()
    final_test_start = bundle.daily_sales["date"].max() - pd.Timedelta(days=27)
    assert not bundle.daily_sales.loc[bundle.daily_sales["date"].ge(final_test_start), "stockout_flag"].any()
```

- [ ] **Step 2: Run the generator test and verify failure**

Run: `python -m pytest tests/data/test_synthetic.py -v`

Expected: FAIL because `generate_scenario` is undefined.

- [ ] **Step 3: Implement fixed catalog, recipes, and stochastic demand**

Define strict Pydantic models `ItemDefinition`, `IngredientDefinition`, `RecipeDefinition`, and `SyntheticScenarioDefinition`; forbid unknown YAML keys. Item fields include ID/name/category/base price/base demand, weekday/month/weather/holiday effects, and launch date. Ingredient fields include ID/name/base unit/waste/pack/lead/review/shelf-life/perishable. Recipe fields contain item ID, ingredient ID, amount, and unit. Put the exact 15 menu items and 15 ingredients from the design spec in `configs/synthetic.yaml`. Extend `tests/conftest.py` with `scenario_definition = load_synthetic_definition(Path("configs/synthetic.yaml"))`. Generate daily demand with a seeded NumPy generator and a negative-binomial draw:

```python
log_mu = (
    item.base_log_demand
    + item.weekday_effect[weekday]
    + item.month_effect[month]
    + item.temperature_beta * (temperature - 28.0)
    + item.rain_beta * rain_indicator
    + item.holiday_beta * is_holiday
    + promo_beta * promo_flag
    + trend + preference_drift + random_shock
)
mu = np.exp(log_mu)
probability = dispersion / (dispersion + mu)
quantity = rng.negative_binomial(dispersion, probability)
```

Apply stockout censoring after latent demand is drawn during train/validation dates. Disable exogenous censoring in the final 28-day test window so policy simulation receives uncensored held-out demand; assert `stockout_flag=False` throughout that window. Keep `latent_quantity` only inside the generator while forming observed rows; only `quantity` enters `daily_sales`, preventing an unavailable real-world feature from reaching models. Create at least two Tết closure days per year, promotions known in advance, three item launch dates, inventory lots with expiry dates, and scheduled receipts. Compute a SHA-256 checksum from canonical CSV bytes and place it in `DataProvenance`.

- [ ] **Step 4: Add CLI and run the complete generated bundle through validation**

The CLI command must call `validate_bundle()` before `save_bundle()`. Add a test invoking Typer’s `CliRunner`, then assert all eight CSV files and `provenance.json` exist.

Run: `python -m pytest tests/data/test_synthetic.py tests/data/test_io.py -v`

Run: `fnb-forecast generate-synthetic --project-config configs/base.yaml --scenario-config configs/synthetic.yaml --output data/processed/synthetic`

Expected: tests PASS; command writes 730 days with provenance `synthetic_vietnam_scenario`.

- [ ] **Step 5: Commit the scenario generator**

```powershell
git add configs/synthetic.yaml src/fnb_forecast/data/synthetic.py src/fnb_forecast/cli.py tests/conftest.py tests/data/test_synthetic.py README.md
git commit -m "feat: generate reproducible Vietnam F&B scenario"
```

### Task 5: M5 Acquisition, Normalization, and Leakage-Safe SKU Selection

**Files:**
- Create: `scripts/download_m5.py`
- Create: `src/fnb_forecast/data/m5.py`
- Create: `tests/data/test_m5.py`
- Create: `data/README.md`
- Modify: `src/fnb_forecast/cli.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `DataBundle`, table validators, official M5 CSV files.
- Produces: `download_m5(output_dir: Path) -> Path`, `load_m5(raw_dir: Path, store_id: str, n_items: int, selection_end: pd.Timestamp) -> DataBundle`, `select_food_items(sales: pd.DataFrame, selection_end: pd.Timestamp, n_items: int) -> list[str]`, and CLI `fnb-forecast prepare-m5`.

- [ ] **Step 1: Write a test proving selection cannot inspect future rows**

```python
# tests/data/test_m5.py
import pandas as pd
from fnb_forecast.data.m5 import select_food_items

def test_item_selection_uses_only_dates_before_selection_end() -> None:
    sales = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
        "item_id": ["stable", "stable", "future_spike", "future_spike"],
        "category": ["FOODS"] * 4,
        "quantity": [5, 5, 1000, 1000],
    })
    selected = select_food_items(sales, pd.Timestamp("2024-01-15"), n_items=1)
    assert selected == ["stable"]
```

- [ ] **Step 2: Run the M5 test and verify failure**

Run: `python -m pytest tests/data/test_m5.py -v`

Expected: FAIL because the M5 module does not exist.

- [ ] **Step 3: Implement download/checksum and wide-to-long normalization**

`scripts/download_m5.py` must call the official Kaggle CLI without embedding credentials:

```python
subprocess.run(
    [sys.executable, "-m", "kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy", "-p", str(output_dir)],
    check=True,
)
```

Unzip only under the explicit output directory, calculate SHA-256 for `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`, and write `checksums.json`. `load_m5()` must:

1. filter one configured store, default `CA_1`;
2. melt `d_1...d_1969` into daily long form;
3. join `calendar.date` and weekly price;
4. restrict candidates to `cat_id == "FOODS"`;
5. rank by non-zero-day count, then total units, then `item_id`, using rows `date <= selection_end` only;
6. keep exactly 20 items for the real run;
7. build a `DataBundle` with empty but schema-valid recipe/inventory tables and provenance `m5_public_benchmark`.

- [ ] **Step 4: Add normalization fixture and CLI tests**

Create tiny wide M5 CSV fixtures under `tests/fixtures/m5/`. Assert `prepare-m5` outputs sorted `(item_id,date)` rows, price joins correctly, and no recipe/inventory values are fabricated.

Run: `python -m pytest tests/data/test_m5.py -v`

Expected: PASS using only fixture files; no Kaggle network call occurs.

- [ ] **Step 5: Document acquisition and commit**

In `data/README.md`, include the official competition URL, Kaggle authentication prerequisite, exact CLI commands, expected filenames, and the rule that raw files/checksums generated from local downloads are not committed.

```powershell
git add scripts/download_m5.py src/fnb_forecast/data/m5.py src/fnb_forecast/cli.py tests/data/test_m5.py tests/fixtures/m5 data/README.md .gitignore
git commit -m "feat: prepare leakage-safe M5 benchmark subset"
```

### Task 6: Temporal Splits and As-Of Feature Pipeline

**Files:**
- Create: `src/fnb_forecast/features/__init__.py`
- Create: `src/fnb_forecast/features/builder.py`
- Create: `src/fnb_forecast/evaluation/__init__.py`
- Create: `src/fnb_forecast/evaluation/splits.py`
- Test: `tests/features/test_builder.py`
- Test: `tests/evaluation/test_splits.py`

**Interfaces:**
- Consumes: validated `daily_sales`, `calendar`, and `weather` frames.
- Produces: `ForecastFold`, `make_evaluation_splits(dates: pd.DatetimeIndex, horizon_days: int = 7, validation_folds: int = 8, test_days: int = 28) -> tuple[list[ForecastFold], list[ForecastFold]]`, and `FeatureBuilder.build(history, future_known, origin) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write split and lag leakage tests**

```python
# tests/evaluation/test_splits.py
import pandas as pd
from fnb_forecast.evaluation.splits import make_evaluation_splits

def test_splits_reserve_final_28_days_and_have_seven_day_horizons() -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    validation, test = make_evaluation_splits(dates)
    assert len(validation) == 8
    assert len(test) == 4
    assert all(len(fold.target_dates) == 7 for fold in validation + test)
    assert max(date for fold in validation for date in fold.target_dates) < min(date for fold in test for date in fold.target_dates)
    assert max(date for fold in test for date in fold.target_dates) == dates.max()
```

```python
# tests/features/test_builder.py
def test_lag_and_rolling_features_use_only_prior_days():
    sales = make_single_item_sales([1, 2, 3, 4, 5, 6, 7, 100])
    train, _ = FeatureBuilder().build(sales.iloc[:7], make_known_days(7), pd.Timestamp("2024-01-07"))
    last = train.iloc[-1]
    assert last["lag_1"] == 6
    assert last["rolling_mean_7"] == 3.5
    assert 100 not in train.select_dtypes("number").to_numpy()
```

- [ ] **Step 2: Run tests and verify missing split/feature modules**

Run: `python -m pytest tests/features/test_builder.py tests/evaluation/test_splits.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Implement frozen fold objects and feature generation**

Use this immutable split contract:

```python
@dataclass(frozen=True)
class ForecastFold:
    name: str
    forecast_origin: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    target_dates: tuple[pd.Timestamp, ...]
    role: Literal["validation", "test"]
```

`FeatureBuilder` must create lags 1, 7, 14, 21, 28 and rolling mean/std/min/max for 7, 14, 28 days using `groupby("item_id")["quantity"].shift(1)` before `rolling`. Join calendar on `target_date`. Join weather on both `target_date` and the latest `issued_at <= origin`; raise `WeatherUnavailableError` if no valid issue exists. Fit category encoders and numeric scalers only through `fit(history)` and serialize their learned state.

- [ ] **Step 4: Add boundary tests for future price/promo and weather**

Assert future price/promotion rows are accepted only for the seven target dates, target quantity is absent from `future_known`, and a weather row issued one minute after the origin is rejected.

Run: `python -m pytest tests/features tests/evaluation/test_splits.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit temporal feature infrastructure**

```powershell
git add src/fnb_forecast/features src/fnb_forecast/evaluation tests/features tests/evaluation/test_splits.py
git commit -m "feat: build leakage-safe temporal features and splits"
```

### Task 7: Metrics, Bootstrap Confidence Intervals, and Backtest Engine

**Files:**
- Create: `src/fnb_forecast/models/__init__.py`
- Create: `src/fnb_forecast/models/base.py`
- Create: `src/fnb_forecast/evaluation/metrics.py`
- Create: `src/fnb_forecast/evaluation/backtest.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_backtest.py`

**Interfaces:**
- Consumes: `ForecastModel`, `ForecastFold`, `FeatureBuilder`, `DataBundle`.
- Produces: `wape`, `mae`, `rmsse`, `forecast_bias`, `paired_bootstrap_wape_difference`, `BacktestRunner.run(model_factory, bundle, folds) -> pd.DataFrame`.

- [ ] **Step 1: Write exact metric tests**

```python
# tests/evaluation/test_metrics.py
import numpy as np
from fnb_forecast.evaluation.metrics import wape, mae, rmsse, forecast_bias

def test_metrics_match_hand_calculation() -> None:
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([2.0, 8.0, 25.0])
    assert wape(actual, predicted) == 9 / 30
    assert mae(actual, predicted) == 3
    assert forecast_bias(actual, predicted) == 5 / 30
    assert np.isclose(rmsse(actual, predicted, np.array([1, 2, 3, 4, 5, 6, 7, 8]), season_length=7), np.sqrt(11 / 49))

def test_wape_zero_denominator_returns_nan() -> None:
    assert np.isnan(wape(np.zeros(3), np.ones(3)))
```

- [ ] **Step 2: Run metric tests and verify failure**

Run: `python -m pytest tests/evaluation/test_metrics.py -v`

Expected: FAIL because metric functions are missing.

- [ ] **Step 3: Implement metrics and a spy-model backtest**

Use these formulas:

```python
def wape(actual, predicted):
    denominator = np.abs(actual).sum()
    return np.nan if denominator == 0 else np.abs(actual - predicted).sum() / denominator

def forecast_bias(actual, predicted):
    denominator = actual.sum()
    return np.nan if denominator == 0 else (predicted - actual).sum() / denominator

def rmsse(actual, predicted, training, season_length=7):
    scale = np.mean((training[season_length:] - training[:-season_length]) ** 2)
    return np.nan if scale == 0 else np.sqrt(np.mean((actual - predicted) ** 2) / scale)
```

`BacktestRunner` must instantiate a fresh model per fold/config, slice `history.date <= forecast_origin`, pass only the seven future-known rows per item, assert output uniqueness on `(forecast_origin,target_date,item_id)`, join actuals after prediction, and write `fold_name`, `role`, and `dataset_name` columns.

- [ ] **Step 4: Prove the runner never exposes future targets**

Create `SpyModel` in `tests/evaluation/test_backtest.py`; its `fit()` asserts `history.date.max() <= origin`, and its `predict()` asserts `quantity` is absent from `future_covariates`. Run two folds and assert output contains `7 × item_count × 2` rows.

Implement paired bootstrap with `np.random.default_rng(seed)` sampling unique `(item_id,fold_name)` blocks 2,000 times and returning `mean_difference`, `lower_95`, `upper_95`.

Run: `python -m pytest tests/evaluation -v`

Expected: all split, metric, and backtest tests PASS.

- [ ] **Step 5: Commit evaluation core**

```powershell
git add src/fnb_forecast/models/base.py src/fnb_forecast/evaluation tests/evaluation
git commit -m "feat: add rolling backtests and forecasting metrics"
```

### Task 8: Seasonal Naive and AutoETS Baselines

**Files:**
- Create: `src/fnb_forecast/models/naive.py`
- Create: `src/fnb_forecast/models/ets.py`
- Test: `tests/models/test_naive.py`
- Test: `tests/models/test_ets.py`
- Modify: `src/fnb_forecast/models/__init__.py`

**Interfaces:**
- Consumes: `ForecastModel` contract and history/future frames from Task 6.
- Produces: `SeasonalNaiveModel(season_length: int = 7)` and `ETSModel(season_length: int = 7)`.

- [ ] **Step 1: Write exact seasonal-naive behavior tests**

```python
# tests/models/test_naive.py
def test_seasonal_naive_repeats_previous_week(make_model_frames):
    history, future = make_model_frames(values=list(range(1, 15)), horizon=7)
    model = SeasonalNaiveModel(season_length=7)
    model.fit(history, future.iloc[0:0])
    result = model.predict(history, future)
    assert result["yhat"].tolist() == list(range(8, 15))
    assert result["model_id"].unique().tolist() == ["seasonal_naive_7"]
```

- [ ] **Step 2: Run baseline tests and verify failure**

Run: `python -m pytest tests/models/test_naive.py tests/models/test_ets.py -v`

Expected: FAIL because the adapters do not exist.

- [ ] **Step 3: Implement both adapters behind the same protocol**

Seasonal Naive must use exact `target_date - 7 days`; if a value is unavailable, use the item’s non-null training median and add `fallback_reason="missing_seasonal_lag"` to adapter metadata. ETS must reshape history to StatsForecast columns `unique_id`, `ds`, `y`, fit `AutoETS(season_length=7, model="ZZZ")`, forecast seven days, map back to item IDs, and clamp numeric negatives to zero.

```python
forecast = self._statsforecast.forecast(h=7)
forecast["yhat"] = forecast["AutoETS"].clip(lower=0.0)
```

Both adapters save a versioned `metadata.json`; ETS also saves its fitted object with `joblib`.

- [ ] **Step 4: Run adapters through the shared backtester**

Add a parametrized test over both adapters using two items and two folds. Assert five required forecast columns, no duplicates, no negative predictions, and exact seven dates.

Run: `python -m pytest tests/models/test_naive.py tests/models/test_ets.py tests/evaluation/test_backtest.py -v`

Expected: PASS for both adapters.

- [ ] **Step 5: Commit statistical baselines**

```powershell
git add src/fnb_forecast/models tests/models
git commit -m "feat: add seasonal naive and ETS baselines"
```

### Task 9: Global Horizon-Index LightGBM

**Files:**
- Create: `configs/models/lightgbm.yaml`
- Create: `src/fnb_forecast/models/lightgbm.py`
- Test: `tests/models/test_lightgbm.py`
- Modify: `src/fnb_forecast/models/__init__.py`

**Interfaces:**
- Consumes: feature frames from `FeatureBuilder`, `ForecastModel` contract.
- Produces: `LightGBMForecastModel(config: LightGBMConfig)` and `build_horizon_training_table(history: pd.DataFrame, max_horizon: int = 7) -> pd.DataFrame`.

`LightGBMConfig` fields are `name`, `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `subsample`, `subsample_freq`, `colsample_bytree`, and `seed`; forbid extra fields.

- [ ] **Step 1: Write horizon-table and deterministic prediction tests**

```python
# tests/models/test_lightgbm.py
def test_training_table_has_one_row_per_origin_item_horizon(feature_history):
    table = build_horizon_training_table(feature_history, max_horizon=7)
    assert set(table["horizon"].unique()) == set(range(1, 8))
    assert (table["target_date"] - table["forecast_origin"]).dt.days.equals(table["horizon"])
    assert table.groupby(["item_id", "forecast_origin", "horizon"]).size().eq(1).all()

def test_lightgbm_predictions_are_repeatable_and_nonnegative(model_frames):
    first = LightGBMForecastModel(test_lightgbm_config(seed=17))
    first.fit(*model_frames.fit_args)
    p1 = first.predict(*model_frames.predict_args)
    second = LightGBMForecastModel(test_lightgbm_config(seed=17))
    second.fit(*model_frames.fit_args)
    p2 = second.predict(*model_frames.predict_args)
    assert p1["yhat"].ge(0).all()
    assert p1.equals(p2)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/models/test_lightgbm.py -v`

Expected: FAIL because the model and table builder do not exist.

- [ ] **Step 3: Implement one global model with horizon as a feature**

Build training origins only where all seven future targets exist and are still inside the supplied history. Join the target from `origin + horizon`; never calculate lag/rolling features relative to `target_date`. Encode item/category as pandas categorical codes fitted from training. Use one `LGBMRegressor`:

```python
self.estimator = LGBMRegressor(
    objective="regression_l1",
    n_estimators=config.n_estimators,
    learning_rate=config.learning_rate,
    num_leaves=config.num_leaves,
    max_depth=config.max_depth,
    subsample=config.subsample,
    colsample_bytree=config.colsample_bytree,
    random_state=config.seed,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)
```

`configs/models/lightgbm.yaml` must contain exactly 12 named candidates varying `num_leaves`, `max_depth`, `learning_rate`, and `n_estimators`; all set seed `20260813` and CPU threads to `-1`. `predict()` builds seven rows per item, applies `np.maximum(raw, 0.0)`, and returns the shared schema.

Use the exact Cartesian grid `num_leaves=[15,31,63] × max_depth=[-1,8] × learning_rate=[0.03,0.05]`. Set `n_estimators=800` for learning rate `0.03`, otherwise `500`; set `subsample=0.9`, `subsample_freq=1`, and `colsample_bytree=0.9`. Name candidates `lgb_l{leaves}_d{depth}_lr{rate}` and assert the loaded list has length 12 with unique names.

- [ ] **Step 4: Add save/load compatibility test**

Fit on a small fixture, save estimator, encoders, feature-order list, package version, training checksum, and config to an artifact directory. Reload and assert prediction equality and rejection when a required feature is missing.

Run: `python -m pytest tests/models/test_lightgbm.py tests/evaluation/test_backtest.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit LightGBM**

```powershell
git add configs/models/lightgbm.yaml src/fnb_forecast/models/lightgbm.py src/fnb_forecast/models/__init__.py tests/models/test_lightgbm.py
git commit -m "feat: add global multi-horizon LightGBM"
```

### Task 10: Global Direct Multi-Output LSTM

**Files:**
- Create: `configs/models/lstm.yaml`
- Create: `src/fnb_forecast/models/lstm.py`
- Test: `tests/models/test_lstm.py`
- Modify: `src/fnb_forecast/models/__init__.py`

**Interfaces:**
- Consumes: normalized feature history, future-known covariates, shared model contract.
- Produces: `SequenceExample`, `LSTMWindowDataset`, `GlobalLSTM(nn.Module)`, `LSTMForecastModel(config: LSTMConfig)`.

`LSTMConfig` fields are `name`, `input_window` (`28|56`), `hidden_size`, `num_layers`, `dropout`, `batch_size`, `learning_rate`, `weight_decay`, `max_epochs`, `early_stopping_patience`, and `seed`; forbid extra fields.

```python
@dataclass(frozen=True)
class SequenceExample:
    history_numeric: torch.Tensor
    future_known: torch.Tensor
    item_id: torch.Tensor
    category_id: torch.Tensor
    target: torch.Tensor
    forecast_origin: pd.Timestamp
    target_dates: tuple[pd.Timestamp, ...]
```

- [ ] **Step 1: Write dataset-window and output-shape tests**

```python
# tests/models/test_lstm.py
import torch

def test_window_dataset_stops_targets_at_origin_plus_seven(lstm_frame):
    dataset = LSTMWindowDataset(lstm_frame, input_window=28, horizon=7)
    sample = dataset[0]
    assert sample.history_numeric.shape[0] == 28
    assert sample.future_known.shape[0] == 7
    assert sample.target.shape == (7,)
    assert sample.target_dates.min() > sample.forecast_origin

def test_global_lstm_outputs_nonnegative_seven_day_vector():
    model = GlobalLSTM(history_features=6, future_features=4, item_count=15, category_count=4, hidden_size=16, num_layers=1, dropout=0.0)
    output = model(make_lstm_batch(batch_size=3))
    assert output.shape == (3, 7)
    assert torch.all(output >= 0)
```

- [ ] **Step 2: Run LSTM tests and verify failure**

Run: `python -m pytest tests/models/test_lstm.py -v`

Expected: FAIL because LSTM classes do not exist.

- [ ] **Step 3: Implement dataset and model**

The dataset yields 28/56 past rows, seven known-future rows, item/category indices, seven targets, origin, and target dates. Exclude closed-store rows from eligible origins. Standardize numeric inputs using training-only mean/std and store those values in adapter state.

Implement the network as:

```python
class GlobalLSTM(nn.Module):
    def __init__(self, history_features, future_features, item_count, category_count, hidden_size, num_layers, dropout):
        super().__init__()
        self.item_embedding = nn.Embedding(item_count, 8)
        self.category_embedding = nn.Embedding(category_count, 4)
        self.encoder = nn.LSTM(history_features + 12, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size + 7 * future_features + 12, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 7), nn.Softplus())

    def forward(self, batch):
        static = torch.cat([self.item_embedding(batch.item_id), self.category_embedding(batch.category_id)], dim=-1)
        repeated = static.unsqueeze(1).expand(-1, batch.history_numeric.size(1), -1)
        _, (hidden, _) = self.encoder(torch.cat([batch.history_numeric, repeated], dim=-1))
        joined = torch.cat([hidden[-1], batch.future_known.flatten(1), static], dim=-1)
        return self.head(joined)
```

Train with `nn.HuberLoss`, AdamW, gradient clipping at 1.0, maximum 100 epochs, and early stopping after 10 validation epochs without WAPE improvement. Seed Python/NumPy/PyTorch and set deterministic algorithms where supported.

Use exactly 12 YAML candidates from `input_window=[28,56] × hidden_size=[32,64,128] × dropout=[0.1,0.3]`. Fix `num_layers=2`, `batch_size=64`, `learning_rate=0.001`, `weight_decay=0.0001`, `max_epochs=100`, and `early_stopping_patience=10`. Name candidates `lstm_w{window}_h{hidden}_d{dropout}` and reject duplicate names.

- [ ] **Step 4: Test tiny-batch learning, determinism, and save/load**

Use a 64-window fixture with a deterministic weekly pattern. Train at most 20 epochs in the test and assert final Huber loss is below initial loss, output is nonnegative, two CPU runs with the same seed agree within `atol=1e-5`, and a reloaded artifact agrees within the same tolerance.

Run: `python -m pytest tests/models/test_lstm.py -v`

Expected: PASS on CPU in under 60 seconds.

- [ ] **Step 5: Commit LSTM**

```powershell
git add configs/models/lstm.yaml src/fnb_forecast/models/lstm.py src/fnb_forecast/models/__init__.py tests/models/test_lstm.py
git commit -m "feat: add global direct multi-output LSTM"
```

### Task 11: Temporal Fusion Transformer Adapter

**Files:**
- Create: `configs/models/tft.yaml`
- Create: `src/fnb_forecast/models/tft.py`
- Test: `tests/models/test_tft.py`
- Modify: `src/fnb_forecast/models/__init__.py`

**Interfaces:**
- Consumes: long feature frame and `ForecastModel` contract.
- Produces: `build_tft_dataset(frame, config, training: bool) -> TimeSeriesDataSet`, `TFTForecastModel(config: TFTConfig)`, `TFTInterpretation` with attention and variable-selection tables.

`TFTConfig` fields are `name`, `input_window` (`28|56`), `hidden_size`, `attention_head_size`, `hidden_continuous_size`, `dropout`, `batch_size`, `learning_rate`, `max_epochs`, `early_stopping_patience`, `gradient_clip_val`, `limit_train_batches`, and `seed`; forbid extra fields.

```python
@dataclass
class TFTInterpretation:
    attention: pd.DataFrame
    variable_importance: pd.DataFrame
    metadata: dict[str, Any]  # always contains causal_interpretation=False
```

- [ ] **Step 1: Write role-assignment and CPU smoke tests**

```python
# tests/models/test_tft.py
def test_tft_dataset_keeps_target_out_of_known_future(tft_frame):
    dataset = build_tft_dataset(tft_frame, test_tft_config(), training=True)
    assert "quantity" in dataset.time_varying_unknown_reals
    assert "quantity" not in dataset.time_varying_known_reals
    assert {"weekday", "month", "promo_flag", "unit_price", "temperature", "rain", "humidity"}.issubset(dataset.time_varying_known_reals)

def test_tft_cpu_smoke_returns_seven_days_per_item(tft_frame):
    adapter = TFTForecastModel(test_tft_config(max_epochs=1, limit_train_batches=2))
    adapter.fit(tft_frame.history, tft_frame.known_covariates)
    result = adapter.predict(tft_frame.history, tft_frame.future_covariates)
    assert result.groupby("item_id").size().eq(7).all()
    assert result["yhat"].ge(0).all()
```

- [ ] **Step 2: Run TFT tests and verify failure**

Run: `python -m pytest tests/models/test_tft.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the stable PyTorch Forecasting 1.x adapter**

Build `TimeSeriesDataSet` with `group_ids=["item_id"]`, `time_idx`, target `quantity`, min/max encoder length equal to the selected `input_window` in `{28,56}`, and min/max prediction length equal to 7. Use item/category as static categoricals, calendar/price/promo/weather as known future variables, and quantity/stockout as unknown historical variables. Fit encoders/normalizers on training only; validation datasets must use `TimeSeriesDataSet.from_dataset(training_dataset, validation_frame, stop_randomization=True)`.

```python
self.model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=config.learning_rate,
    hidden_size=config.hidden_size,
    attention_head_size=config.attention_head_size,
    hidden_continuous_size=config.hidden_continuous_size,
    dropout=config.dropout,
    output_size=1,
    loss=MAE(),
    reduce_on_plateau_patience=4,
)
```

Train with Lightning CPU/GPU auto-selection, maximum 100 epochs, early-stopping patience 10, deterministic seed, and gradient clipping 1.0. Clamp point predictions to zero. Save the Lightning checkpoint plus dataset parameters, package versions, config, and feature-order metadata.

Use exactly 12 YAML candidates from `input_window=[28,56] × hidden_size=[16,32,64] × attention_head_size=[1,4]`. Set `hidden_continuous_size=min(hidden_size//2,16)`, `dropout=0.1`, `batch_size=64`, `learning_rate=0.001`, `max_epochs=100`, `early_stopping_patience=10`, `gradient_clip_val=1.0`, and full-profile `limit_train_batches=1.0`. Name candidates `tft_w{window}_h{hidden}_a{heads}`.

- [ ] **Step 4: Add interpretation disclaimer and artifact round trip**

`interpret()` returns normalized attention and variable-selection weights with metadata field `causal_interpretation=False`. Test that dashboard-ready interpretation never labels a variable as “cause”, and that save/load preserves forecasts within `atol=1e-5` on CPU.

Run: `python -m pytest tests/models/test_tft.py -v`

Expected: PASS; the smoke test is bounded to two train batches.

- [ ] **Step 5: Commit TFT**

```powershell
git add configs/models/tft.yaml src/fnb_forecast/models/tft.py src/fnb_forecast/models/__init__.py tests/models/test_tft.py
git commit -m "feat: add temporal fusion transformer adapter"
```

### Task 12: Experiment Orchestration, Model Selection, Registry, and Intervals

**Files:**
- Create: `src/fnb_forecast/models/registry.py`
- Create: `src/fnb_forecast/evaluation/experiment.py`
- Create: `src/fnb_forecast/evaluation/uncertainty.py`
- Create: `tests/evaluation/test_experiment.py`
- Create: `tests/evaluation/test_uncertainty.py`
- Create: `tests/models/test_registry.py`
- Modify: `src/fnb_forecast/cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all five model adapters, validation/test folds, metrics, checksums.
- Produces: `ExperimentRunner.run(dataset_name, bundle, config) -> ExperimentResult`, `select_model(leaderboard: pd.DataFrame) -> str`, `ResidualIntervalCalibrator.fit(validation_predictions)`, `ModelRegistry.save/load`, CLI commands `train`, `evaluate`, and `report`.

```python
@dataclass
class ExperimentResult:
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    leaderboard: pd.DataFrame
    ablation_metrics: pd.DataFrame
    selected_model_id: str
    selected_run_id: str
    metadata: dict[str, Any]
```

- [ ] **Step 1: Write selector and residual-interval tests**

```python
# tests/evaluation/test_experiment.py
def test_selector_prefers_simpler_model_inside_one_percent_tie():
    board = pd.DataFrame({
        "model_id": ["lstm", "lightgbm"],
        "validation_wape": [0.199, 0.200],
        "parameter_count": [50000, 500],
        "inference_ms": [30.0, 4.0],
    })
    assert select_model(board) == "lightgbm"

def test_selector_is_run_independently_per_dataset():
    board = make_two_dataset_leaderboard(m5_best="ets", synthetic_best="lstm")
    assert select_models_by_dataset(board) == {"m5_public_benchmark": "ets", "synthetic_vietnam_scenario": "lstm"}
```

```python
# tests/evaluation/test_uncertainty.py
def test_interval_uses_only_validation_residuals():
    calibrator = ResidualIntervalCalibrator(coverage=0.80)
    calibrator.fit(make_residuals(role="validation"))
    with pytest.raises(ValueError, match="validation"):
        calibrator.fit(make_residuals(role="test"))
```

- [ ] **Step 2: Run orchestration tests and verify failure**

Run: `python -m pytest tests/evaluation/test_experiment.py tests/evaluation/test_uncertainty.py tests/models/test_registry.py -v`

Expected: FAIL because orchestrator, interval calibrator, and registry do not exist.

- [ ] **Step 3: Implement bounded grids, three seeds, and ablations**

Load exactly 12 candidates for LightGBM, LSTM, and TFT. Run each LSTM/TFT candidate with seeds `20260813`, `20260814`, `20260815`; aggregate validation metrics by mean and standard deviation. Baselines run once. Execute these ablations:

```python
SYNTHETIC_ABLATIONS = {
    "history": HISTORY_FEATURES,
    "history_calendar": HISTORY_FEATURES + CALENDAR_FEATURES,
    "full": HISTORY_FEATURES + CALENDAR_FEATURES + PRICE_PROMO_FEATURES + WEATHER_FEATURES,
}
M5_ABLATIONS = {
    "history": HISTORY_FEATURES,
    "history_calendar": HISTORY_FEATURES + M5_CALENDAR_FEATURES,
    "full": HISTORY_FEATURES + M5_CALENDAR_FEATURES + ["unit_price"],
}
```

Select by mean validation WAPE. Treat models within a relative 1% of the minimum as tied, then sort by `parameter_count`, `inference_ms`, and `model_id`. After selection, evaluate the chosen model once on four untouched test origins. Do not add test metrics back into the selector input.

For the synthetic dataset, register two deployable choices: the overall selected `full` artifact and the best `history_calendar` artifact as the explicit no-weather fallback. If an archived weather forecast is unavailable for a validation fold, skip only the `full` run for that fold with a recorded reason; never substitute observed weather. The `history` and `history_calendar` ablations must still complete all folds.

- [ ] **Step 4: Implement residual intervals and registry compatibility**

For each `(category,horizon)`, calculate the 10th and 90th percentile validation residuals `actual-yhat`; fall back to horizon-only quantiles when fewer than 20 samples exist. Apply:

```python
lower = np.maximum(0.0, yhat + residual_q10)
upper = np.maximum(lower, yhat + residual_q90)
```

Registry layout:

```text
artifacts/registry/<dataset_name>/<model_id>/<run_id>/
  model files
  feature_state.json
  interval_state.json
  validation_predictions.parquet
  metadata.json
```

`validation_predictions.parquet` contains only out-of-sample validation rows with actual and predicted quantity; it is later converted to ingredient residuals for safety stock. `metadata.json` must include dataset checksum, provenance, config hash, Git commit, Python/package versions, train cutoff, features, seed(s), and schema version. `load()` raises `ArtifactCompatibilityError` on schema/config mismatch; callers decide fallback. Expose `load_validation_predictions(dataset_name: str, model_id: str) -> pd.DataFrame` and reject any row whose role is not `validation`.

Persist validation predictions and residual interval state for every model eligible for inference, including `seasonal_naive_7`; this guarantees safety-stock calculation still works after an explicit runtime fallback.

- [ ] **Step 5: Add CLI integration and test-only fast profile**

CLI examples must be exact:

```powershell
fnb-forecast train --dataset synthetic --config configs/base.yaml --profile test
fnb-forecast evaluate --dataset synthetic --role validation
fnb-forecast evaluate --dataset synthetic --role test --selected-only
fnb-forecast report --dataset synthetic --output artifacts/reports/synthetic
```

The `test` profile uses one candidate, one seed, one epoch/two batches but follows the identical code path. Run:

`python -m pytest tests/evaluation/test_experiment.py tests/evaluation/test_uncertainty.py tests/models/test_registry.py -v`

Expected: PASS; tests confirm no test-row access before selection.

- [ ] **Step 6: Commit experiment orchestration**

```powershell
git add src/fnb_forecast/models/registry.py src/fnb_forecast/evaluation src/fnb_forecast/cli.py tests/evaluation tests/models/test_registry.py README.md
git commit -m "feat: orchestrate reproducible model experiments"
```

### Task 13: Revenue and BOM Conversion

**Files:**
- Create: `src/fnb_forecast/planning/__init__.py`
- Create: `src/fnb_forecast/planning/revenue.py`
- Test: `tests/planning/test_revenue.py`

**Interfaces:**
- Consumes: forecast frame, future item prices, `recipes`, `ingredient_master`.
- Produces: `IngredientNeedResult`, `PlanningOutput`, `calculate_revenue(forecasts, prices) -> pd.DataFrame`, `calculate_ingredient_need(forecasts, recipes) -> IngredientNeedResult`.

```python
@dataclass
class IngredientNeedResult:
    totals: pd.DataFrame
    detail: pd.DataFrame
    warnings: list[str]

@dataclass
class PlanningOutput:
    item_forecasts: pd.DataFrame
    revenue_forecast: pd.DataFrame
    ingredient_forecast: pd.DataFrame
    ingredient_detail: pd.DataFrame
    order_proposals: pd.DataFrame
    warnings: list[str]
    metadata: dict[str, Any]
```

- [ ] **Step 1: Write exact revenue, waste, and missing-BOM tests**

```python
# tests/planning/test_revenue.py
def test_revenue_and_bom_conversion_are_consistent():
    forecasts = pd.DataFrame({"forecast_origin": ["2025-01-01"], "target_date": ["2025-01-02"], "item_id": ["milk_tea"], "yhat": [10.0], "model_id": ["lstm"]})
    prices = pd.DataFrame({"target_date": ["2025-01-02"], "item_id": ["milk_tea"], "unit_price": [30000]})
    recipes = pd.DataFrame({"item_id": ["milk_tea", "milk_tea"], "ingredient_id": ["milk", "tea"], "amount": [120.0, 8.0], "unit": ["ml", "g"], "waste_rate": [0.05, 0.10]})
    revenue = calculate_revenue(forecasts, prices)
    ingredients = calculate_ingredient_need(forecasts, recipes)
    assert revenue["forecast_revenue"].item() == 300000
    assert ingredients.totals.set_index("ingredient_id").loc["milk", "forecast_need"] == 1260
    assert ingredients.totals.set_index("ingredient_id").loc["tea", "forecast_need"] == 88
    assert ingredients.warnings == []

def test_missing_bom_warns_and_excludes_only_that_item():
    ingredients = calculate_ingredient_need(two_item_forecast(), one_item_recipe())
    assert ingredients.warnings == ["Missing BOM for item: unknown_item"]
    assert "unknown_item" not in ingredients.detail["source_item_id"].unique()
```

- [ ] **Step 2: Run planning tests and verify failure**

Run: `python -m pytest tests/planning/test_revenue.py -v`

Expected: FAIL because planning functions do not exist.

- [ ] **Step 3: Implement vectorized, unit-safe conversion**

Validate price uniqueness on `(target_date,item_id)` and BOM uniqueness on `(item_id,ingredient_id)`. Merge with `validate="one_to_one"` for price and `validate="many_to_many"` for recipes. Reject missing prices because revenue would be incomplete. Preserve missing BOM as an explicit warning and exclude only those items from ingredient totals.

```python
merged["forecast_revenue"] = merged["yhat"] * merged["unit_price"]
bom["forecast_need"] = bom["yhat"] * bom["amount"] * (1.0 + bom["waste_rate"])
```

Aggregate revenue by date and ingredient need by `(target_date,ingredient_id,unit)` into `IngredientNeedResult.totals`. Keep an auditable frame by `(target_date,ingredient_id,source_item_id)` in `IngredientNeedResult.detail` for the dashboard drill-down.

- [ ] **Step 4: Add schema/duplicate/unit failure tests**

Assert duplicate prices, mixed base units for one ingredient, negative waste rate, and waste rate above 1.0 raise `DataValidationError` with table and offending row indices.

Run: `python -m pytest tests/planning/test_revenue.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit revenue and BOM planning**

```powershell
git add src/fnb_forecast/planning tests/planning/test_revenue.py
git commit -m "feat: convert item forecasts to revenue and ingredients"
```

### Task 14: FEFO Ordering Policy and Inventory Simulation

**Files:**
- Create: `src/fnb_forecast/planning/inventory.py`
- Create: `src/fnb_forecast/planning/simulation.py`
- Test: `tests/planning/test_inventory.py`
- Test: `tests/planning/test_simulation.py`
- Modify: `src/fnb_forecast/cli.py`

**Interfaces:**
- Consumes: validation item predictions, recipes, ingredient forecast, lots, scheduled receipts, ingredient master.
- Produces: `build_ingredient_residuals(validation_predictions, recipes) -> pd.DataFrame`, `compute_safety_stock(residuals, ingredient_master, service_level) -> pd.DataFrame`, `suggest_orders(...) -> pd.DataFrame`, `allocate_fefo(...)`, `InventorySimulator.run(...) -> SimulationResult`, CLI `simulate-inventory`.

```python
@dataclass
class SimulationResult:
    policy_summary: pd.DataFrame
    daily_audit: pd.DataFrame
    metadata: dict[str, Any]
```

- [ ] **Step 1: Write safety-stock, expiry, and pack-rounding tests**

```python
# tests/planning/test_inventory.py
import numpy as np

def test_safety_stock_uses_underforecast_validation_errors_only():
    residuals = pd.DataFrame({
        "role": ["validation"] * 4,
        "ingredient_id": ["milk"] * 4,
        "target_date": pd.date_range("2025-01-01", periods=4),
        "actual_need": [10, 14, 9, 20],
        "forecast_need": [12, 10, 9, 15],
    })
    master = ingredient_master(lead_time_days=1, review_period_days=1)
    result = compute_safety_stock(residuals, master, service_level=0.95)
    assert np.isclose(result["safety_stock"].item(), 4.9)

def test_order_excludes_expiring_stock_and_rounds_pack_size():
    need = ingredient_need(total=22, ingredient_id="milk")
    master = ingredient_master(pack_size=6, lead_time_days=1, review_period_days=1, shelf_life_days=3)
    lots = inventory_lots(usable=3, expires_before_use=10)
    receipts = scheduled_receipts(quantity=5)
    result = suggest_orders(need, safety_stock_frame(4), lots, receipts, master, order_date=pd.Timestamp("2025-01-01"))
    assert result["usable_on_hand"].item() == 3
    assert result["raw_order"].item() == 18
    assert result["suggested_order"].item() == 18

def test_missing_perishable_shelf_life_disables_cap_with_warning():
    result = suggest_orders(ingredient_need(total=12), safety_stock_frame(2), inventory_lots(usable=0), scheduled_receipts(quantity=0), perishable_master(shelf_life_days=None), order_date=pd.Timestamp("2025-01-01"))
    assert not bool(result["shelf_life_cap_applied"].item())
    assert "Missing shelf life for perishable ingredient" in result["warning"].item()
```

- [ ] **Step 2: Run inventory tests and verify failure**

Run: `python -m pytest tests/planning/test_inventory.py tests/planning/test_simulation.py -v`

Expected: FAIL because inventory modules do not exist.

- [ ] **Step 3: Implement safety stock and FEFO order proposal**

`build_ingredient_residuals()` converts both `actual` and `yhat` from out-of-sample validation predictions through the same BOM, then emits daily `actual_need` and `forecast_need` per ingredient. Reject prediction/residual frames containing any role other than `validation`. Compute daily underforecast `max(actual_need-forecast_need,0)`, full rolling sums over `lead_time_days + review_period_days`, then the configured empirical quantile `0.90`, `0.95`, or `0.98`. Store both requested service level and sample count.

For each lot, calculate whether it remains usable on the projected consumption date; allocate consumption in ascending `(expiry_date,received_date,lot_id)` order. Order formula:

```python
target_stock = coverage_forecast_need + safety_stock
raw_order = max(0.0, target_stock - usable_on_hand - scheduled_receipts)
rounded_order = math.ceil(raw_order / pack_size) * pack_size if raw_order > 0 else 0.0
```

For perishables, calculate the shelf-life cap from forecast need over `shelf_life_days` plus safety stock. If one pack exceeds the remaining cap, keep one pack and set `waste_risk_warning=True`; never silently round down to zero when raw order is positive.

- [ ] **Step 4: Implement day-by-day policy simulation**

`InventorySimulator` must advance in this order: receive scheduled orders, expire lots, create today’s order using only information known today, satisfy actual need via FEFO, record unmet demand, and age remaining lots. Run both policies from deep copies of identical initial state:

```python
forecast_policy = ForecastOrderingPolicy(model_forecasts)
baseline_policy = MovingAverageOrderingPolicy(window_days=7)
```

Return `fill_rate`, `stockout_days`, `waste_rate`, `average_inventory`, and `total_ordered`, plus daily audit rows. Add a hand-calculated three-day fixture proving both policies start identically and that expired stock never satisfies demand.

- [ ] **Step 5: Add CLI test and run planning suite**

CLI:

```powershell
fnb-forecast simulate-inventory --dataset synthetic --service-level 0.95 --output artifacts/reports/inventory
```

Assert unsupported service level `0.92` is rejected and the output contains side-by-side policy metrics and daily audit CSV.

Run: `python -m pytest tests/planning -v`

Expected: all planning and simulation tests PASS.

- [ ] **Step 6: Commit inventory planning**

```powershell
git add src/fnb_forecast/planning src/fnb_forecast/cli.py tests/planning
git commit -m "feat: add FEFO ordering and inventory simulation"
```

### Task 15: Forecast Service, Artifact Fallbacks, and Decision Output

**Files:**
- Create: `src/fnb_forecast/services/__init__.py`
- Create: `src/fnb_forecast/services/forecast.py`
- Test: `tests/services/test_forecast.py`
- Modify: `src/fnb_forecast/contracts.py`
- Modify: `src/fnb_forecast/cli.py`

**Interfaces:**
- Consumes: `ModelRegistry`, `FeatureBuilder`, interval calibrator, revenue/BOM planner, order planner.
- Produces: `ForecastService.run(request: ForecastRequest, bundle: DataBundle, service_level: float) -> PlanningOutput` and CLI `forecast`.

- [ ] **Step 1: Write explicit fallback and warning tests**

```python
# tests/services/test_forecast.py
def test_missing_selected_artifact_falls_back_to_seasonal_naive(service_without_artifact, valid_bundle):
    result = service_without_artifact.run(request(origin="2025-12-24"), valid_bundle, service_level=0.95)
    assert result.item_forecasts["model_id"].unique().tolist() == ["seasonal_naive_7"]
    assert result.item_forecasts["fallback_used"].all()
    assert "Selected artifact unavailable; used seasonal_naive_7" in result.warnings

def test_weather_failure_uses_registered_no_weather_variant(service_with_variants, valid_bundle, failing_weather_client):
    result = service_with_variants.run(request(origin="2025-12-24"), valid_bundle, service_level=0.95)
    assert result.item_forecasts["model_id"].unique().tolist() == ["lstm_history_calendar"]
    assert any("weather" in warning.lower() for warning in result.warnings)
```

- [ ] **Step 2: Run service tests and verify failure**

Run: `python -m pytest tests/services/test_forecast.py -v`

Expected: FAIL because `ForecastService` does not exist.

- [ ] **Step 3: Implement ordered fallback policy and audit metadata**

Use this fixed order:

1. selected full-feature artifact for established items;
2. selected registered no-weather artifact when only weather is unavailable;
3. Seasonal Naive for established items when artifact/schema/inference fails;
4. category median for each new item with fewer than seven historical observations, regardless of the model used for established items.

Do not catch `DataValidationError` from user inputs; return it to the UI. Catch only typed weather/artifact/model inference errors. Attach `fallback_used`, `fallback_reason`, requested model, actual model, origin, dataset checksum, and service level to every run’s metadata and forecast rows.

Call sequence:

```python
bundle = validate_bundle(bundle)
history, future = self.feature_builder.build(...)
predictions = self._predict_with_fallback(history, future, request)
predictions = self.interval_calibrator.apply(predictions)
revenue = calculate_revenue(predictions, future_prices)
ingredient_need = calculate_ingredient_need(predictions, bundle.recipes)
validation_predictions = self.registry.load_validation_predictions(request.dataset_name, run_metadata["established_model_id"])
ingredient_residuals = build_ingredient_residuals(validation_predictions, bundle.recipes)
safety_stock = compute_safety_stock(ingredient_residuals, bundle.ingredient_master, service_level)
orders = suggest_orders(ingredient_need.totals, safety_stock, bundle.inventory_lots, bundle.scheduled_receipts, bundle.ingredient_master, request.forecast_origin)
return PlanningOutput(predictions, revenue, ingredient_need.totals, ingredient_need.detail, orders, warnings + ingredient_need.warnings, run_metadata)
```

- [ ] **Step 4: Test invalid model output and new-item fallback**

Inject a model returning NaN and assert service falls back to Seasonal Naive with the failed model ID recorded. Add a new item with three days of history and assert category median is used only for that item while established items retain the selected model.

Run: `python -m pytest tests/services/test_forecast.py tests/planning -v`

Expected: all service/planning tests PASS.

- [ ] **Step 5: Add CLI decision export and commit**

CLI:

```powershell
fnb-forecast forecast --dataset synthetic --origin 2025-12-24 --service-level 0.95 --output artifacts/forecasts/2025-12-24
```

The command writes `item_forecasts.csv`, `revenue_forecast.csv`, `ingredient_forecast.csv`, `order_proposals.csv`, and `run_metadata.json`.

```powershell
git add src/fnb_forecast/services src/fnb_forecast/contracts.py src/fnb_forecast/cli.py tests/services
git commit -m "feat: orchestrate forecasts with explicit fallbacks"
```

### Task 16: Four-Page Streamlit Dashboard and End-to-End Acceptance

**Files:**
- Create: `src/fnb_forecast/app/__init__.py`
- Create: `src/fnb_forecast/app/Home.py`
- Create: `src/fnb_forecast/app/state.py`
- Create: `src/fnb_forecast/app/components.py`
- Create: `src/fnb_forecast/app/pages/1_Item_Forecast.py`
- Create: `src/fnb_forecast/app/pages/2_Ingredient_Orders.py`
- Create: `src/fnb_forecast/app/pages/3_Model_Evaluation.py`
- Create: `tests/app/test_dashboard.py`
- Create: `tests/integration/test_end_to_end.py`
- Create: `tests/smoke/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ForecastService`, `PlanningOutput`, experiment reports, CSV loader.
- Produces: local Streamlit app with overview, item forecast, ingredient orders, and model evaluation pages; CSV exports; documented end-to-end commands.

- [ ] **Step 1: Write dashboard provenance and four-page smoke tests**

```python
# tests/app/test_dashboard.py
from streamlit.testing.v1 import AppTest

def test_home_displays_synthetic_provenance_and_core_metrics(monkeypatch):
    install_fake_service(monkeypatch, synthetic_planning_output())
    app = AppTest.from_file("src/fnb_forecast/app/Home.py").run()
    assert not app.exception
    assert any("synthetic_vietnam_scenario" in element.value for element in app.markdown)
    assert {metric.label for metric in app.metric} >= {"Doanh thu dự kiến", "Số ly dự kiến", "Cảnh báo kho"}

def test_all_pages_start_without_exception(monkeypatch):
    install_fake_service(monkeypatch, synthetic_planning_output())
    for path in dashboard_page_paths():
        app = AppTest.from_file(path).run()
        assert not app.exception, path
```

- [ ] **Step 2: Run dashboard tests and verify failure**

Run: `python -m pytest tests/app/test_dashboard.py -v`

Expected: FAIL because dashboard files do not exist.

- [ ] **Step 3: Implement shared state and overview page**

`state.py` owns the loaded `DataBundle`, origin, selected service level, and latest `PlanningOutput`. Cache only immutable artifact loading with `st.cache_resource`; do not cache mutable inventory edits. `components.py` provides provenance banner, warning list, VND formatter, interval chart, and CSV download helper.

Home must show:

- provenance banner above all results;
- origin and seven target dates;
- total forecast revenue, total cups, inventory-alert count;
- total demand with 80% interval;
- three highest-priority order/expiry warnings;
- “Chạy dự báo mới” button that calls `ForecastService`, not model training.

- [ ] **Step 4: Implement the three detail pages**

`1_Item_Forecast.py`: item/category filters, actual-versus-forecast Plotly chart, lower/upper ribbon, and forecast table.

`2_Ingredient_Orders.py`: editable current lots, service-level selector limited to `0.90/0.95/0.98`, ingredient demand, expiry warnings, proposal table, and CSV download.

`3_Model_Evaluation.py`: validation/test leaderboard clearly separated, fold chart, ablation table, bias, training/inference times, provenance, and the fixed text “Feature importance/attention is descriptive, not causal.”

Typed `DataValidationError` must render table name, columns, and rows. Fallback warnings must be visible and included in downloads.

- [ ] **Step 5: Add true end-to-end and CLI smoke tests**

```python
# tests/integration/test_end_to_end.py
def test_synthetic_csv_to_decision_exports(tmp_path, test_profile):
    bundle_dir = tmp_path / "bundle"
    generate_synthetic_cli(bundle_dir, test_profile)
    train_test_profile_cli(bundle_dir, tmp_path / "registry")
    output_dir = forecast_cli(bundle_dir, tmp_path / "registry", origin="2025-12-24")
    for name in ["item_forecasts.csv", "revenue_forecast.csv", "ingredient_forecast.csv", "order_proposals.csv", "run_metadata.json"]:
        assert (output_dir / name).exists()
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["provenance"] == "synthetic_vietnam_scenario"
    assert metadata["horizon_days"] == 7
```

Run: `python -m pytest tests/integration/test_end_to_end.py tests/smoke/test_cli.py -v`

Expected: PASS using the bounded test profile on CPU and no internet.

- [ ] **Step 6: Document exact setup, experiment, and demo sequence**

README must contain these commands in order:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
fnb-forecast generate-synthetic --project-config configs/base.yaml --scenario-config configs/synthetic.yaml --output data/processed/synthetic
fnb-forecast train --dataset synthetic --config configs/base.yaml --profile full
fnb-forecast evaluate --dataset synthetic --role validation
fnb-forecast evaluate --dataset synthetic --role test --selected-only
fnb-forecast simulate-inventory --dataset synthetic --service-level 0.95 --output artifacts/reports/inventory
streamlit run src/fnb_forecast/app/Home.py
```

Also document M5 preparation, CPU/GPU behavior, output directories, provenance limitations, and how a POS CSV must match the eight table schemas.

- [ ] **Step 7: Run the complete quality gate**

Run: `python -m ruff check src tests scripts`

Run: `python -m mypy src/fnb_forecast`

Run: `python -m pytest -m "not slow" --cov=src/fnb_forecast --cov-report=term-missing`

Run: `python -m pytest tests/integration/test_end_to_end.py -v`

Expected: lint and type checks report no errors; all non-slow and integration tests PASS; core non-UI modules reach at least 85% line coverage.

- [ ] **Step 8: Commit the dashboard and acceptance flow**

```powershell
git add src/fnb_forecast/app tests/app tests/integration tests/smoke pyproject.toml README.md
git commit -m "feat: deliver F&B forecasting decision dashboard"
```

## Spec Coverage Map

| Design spec sections | Implemented by tasks |
|---|---|
| 1–6: title, goals, research questions, scope, chosen approach | README and experiment/report outputs in Tasks 1, 12, 16 |
| 7: two-layer data strategy | Tasks 3–5 |
| 8: table schemas and provenance | Tasks 1–2 |
| 9: component architecture | Task 1 file boundaries; Tasks 2–16 implement each boundary |
| 10: past/known/static features and leakage rules | Tasks 3 and 6 |
| 11: five forecasting models and bounded tuning | Tasks 8–12 |
| 12: backtesting, metrics, ablation, intervals | Tasks 6, 7, 12 |
| 13: revenue and BOM conversion | Task 13 |
| 14: safety stock, FEFO, ordering, policy simulation | Task 14 |
| 15: four-screen Streamlit dashboard | Task 16 |
| 16: validation, weather/model/new-item/BOM fallbacks | Tasks 2, 3, 8, 12, 14, 15 |
| 17: unit, integration, reproducibility, smoke tests | Every task; final gate in Task 16 |
| 18: focused source tree | File and Interface Map plus Task 1 |
| 19: completion criteria | Task 16 acceptance test and Final Verification Checklist |
| 20: 14-week execution order | Task order 1–16 follows the approved weekly dependency sequence |
| 21: overfit, synthetic validity, leakage, TFT and scope risks | Global Constraints and Tasks 4, 6, 10–12 |
| 22: ethics, local applicability and non-causal wording | Tasks 4, 11, 12, 16 |
| 23: code, data scripts, artifacts, app, tests and docs | Tasks 1–16 |
| 24: future real-store pilot | README limitations in Task 16; no pilot work included |

Self-review result: every approved requirement maps to at least one task; no implementation gap remains.

## Final Verification Checklist

- [ ] Run `git status --short` and confirm only intentionally generated ignored files exist.
- [ ] Run `fnb-forecast report --dataset m5 --output artifacts/reports/m5` and confirm M5 and synthetic leaderboards remain separate.
- [ ] Run the full synthetic workflow and visually inspect all four dashboard pages at 1366×768.
- [ ] Confirm every synthetic chart/table/export contains provenance and every fallback is visible.
- [ ] Confirm final-test results were generated only after model selection metadata was frozen.
- [ ] Compare implementation against all 24 sections of `docs/superpowers/specs/2026-08-13-fnb-demand-revenue-forecasting-design.md` before declaring completion.
