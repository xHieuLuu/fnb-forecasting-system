# Real Store 14-Month Calibrated Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end data ingestion, disaggregation, and 14-month calibrated generative pipeline for real-world cafe/milktea shop data (`dataset.xlsx`), producing a standard 8-table `DataBundle` compatible with model training, evaluation, forecasting, and inventory optimization in `fnb-forecasting`.

**Architecture:** 
1. **Master Data & Parser**: Clean raw `dataset.xlsx`, repair `#DIV/0!` errors, define 43 items, BOM recipes, and ingredient master.
2. **Disaggregation Engine**: Formulate constrained integer sampling matching exact daily revenue ($Y_t$) from anchor months (May & June 2025).
3. **14-Month Temporal Generator**: Synthesize 426-day calibrated series incorporating HCMC weather, Vietnam academic/holiday calendar (Tet 2026, summer, back-to-school, exams), and realistic stockout events.
4. **Bundle Exporter & CLI**: Export standard 8 CSV tables + `provenance.json` to `data/processed/real_store_14m` and integrate with `fnb-forecast` CLI commands.

**Tech Stack:** Python 3.12+, pandas, numpy, scipy/scikit-learn, openpyxl, pydantic, typer/click, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-26-real-store-dataset-design.md`](file:///f:/ugonnaloveit/docs/superpowers/specs/2026-08-26-real-store-dataset-design.md)

## Global Constraints
- Target date range: `2025-05-01` to `2026-06-30` (426 days).
- 43 standard menu items across 6 categories (Coffee, Topping, Fruit tea/Olong/Soda, Iceblend, Milktea, Matcha/Cacao/Dessert).
- Anchor months May & June 2025 daily revenue sums must match `dataset.xlsx` exact totals.
- Output bundle must pass `fnb_forecast.data.validation.validate_bundle` with zero schema/contract violations.

---

### Task 1: Master Data Definitions & Excel Parser

**Files:**
- Create: `src/fnb_forecast/data/real_store_master.py`
- Test: `tests/data/test_real_store_master.py`

**Interfaces:**
- Produces:
  - `get_real_store_item_master() -> pd.DataFrame`
  - `get_real_store_ingredient_master() -> pd.DataFrame`
  - `get_real_store_recipes() -> pd.DataFrame`
  - `parse_real_store_excel(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]` (returns daily revenue series and daily ice expense series)

- [ ] **Step 1: Write failing test for master data and excel parser**

```python
# tests/data/test_real_store_master.py
from pathlib import Path
import pandas as pd
from fnb_forecast.data.real_store_master import (
    get_real_store_item_master,
    get_real_store_ingredient_master,
    get_real_store_recipes,
    parse_real_store_excel,
)

def test_item_master_schema_and_counts():
    items = get_real_store_item_master()
    assert len(items) == 43
    assert set(items.columns) == {"item_id", "item_name", "category", "launch_date", "active"}
    assert items["item_id"].is_unique

def test_ingredient_master_and_recipes_consistency():
    ingredients = get_real_store_ingredient_master()
    recipes = get_real_store_recipes()
    assert set(recipes["ingredient_id"]).issubset(set(ingredients["ingredient_id"]))
    assert set(recipes["item_id"]).issubset(set(get_real_store_item_master()["item_id"]))
    assert (recipes["amount"] > 0).all()
    assert (recipes["waste_rate"] >= 0).all()

def test_parse_real_store_excel():
    excel_path = Path("dataset.xlsx")
    if not excel_path.exists():
        excel_path = Path("f:/ugonnaloveit/dataset.xlsx")
    daily_rev, daily_ice = parse_real_store_excel(excel_path)
    assert not daily_rev.empty
    assert "date" in daily_rev.columns and "revenue" in daily_rev.columns
    # Check May 2025 total is 7572000
    may_rev = daily_rev[daily_rev["date"].dt.month == 5]["revenue"].sum()
    assert may_rev == 7572000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_real_store_master.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing functions.

- [ ] **Step 3: Implement `real_store_master.py`**

Define the 43 items, ingredient master with repaired pack sizes/costs, recipes BOM, and the parser for `dataset.xlsx` extracting May & June daily figures.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_real_store_master.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fnb_forecast/data/real_store_master.py tests/data/test_real_store_master.py
git commit -m "feat(data): add real store master data tables and excel parser"
```

---

### Task 2: Constrained Integer Disaggregation Engine

**Files:**
- Create: `src/fnb_forecast/data/real_store_disaggregation.py`
- Test: `tests/data/test_real_store_disaggregation.py`

**Interfaces:**
- Consumes: `get_real_store_item_master()`, item prices, target daily revenues.
- Produces:
  - `disaggregate_daily_revenue(target_revenue: float, item_prices: dict[str, float], priors: dict[str, float], is_weekend: bool, seed: int = 42) -> dict[str, int]`
  - `disaggregate_revenue_series(daily_rev_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame`

- [ ] **Step 1: Write failing test for disaggregation engine**

```python
# tests/data/test_real_store_disaggregation.py
import pandas as pd
from fnb_forecast.data.real_store_disaggregation import (
    disaggregate_daily_revenue,
    disaggregate_revenue_series,
)
from fnb_forecast.data.real_store_master import get_real_store_item_master

def test_disaggregate_daily_revenue_exact_match():
    items = get_real_store_item_master()
    prices = {row["item_id"]: float(row["unit_price"]) for _, row in items.iterrows()}
    priors = {row["item_id"]: 1.0 for _, row in items.iterrows()}
    
    test_targets = [50000, 200000, 300000, 585000]
    for target in test_targets:
        quantities = disaggregate_daily_revenue(target, prices, priors, is_weekend=False, seed=42)
        total_calc = sum(quantities[item_id] * prices[item_id] for item_id in quantities)
        assert total_calc == target, f"Expected {target}, got {total_calc}"

def test_disaggregate_series():
    df_rev = pd.DataFrame({
        "date": pd.date_range("2025-05-01", periods=5),
        "revenue": [300000, 330000, 450000, 300000, 267000]
    })
    sales_df = disaggregate_revenue_series(df_rev, seed=42)
    assert set(sales_df.columns) >= {"date", "item_id", "quantity", "unit_price"}
    # Verify daily revenue consistency
    for date, group in sales_df.groupby("date"):
        calc_rev = (group["quantity"] * group["unit_price"]).sum()
        expected_rev = df_rev[df_rev["date"] == date]["revenue"].iloc[0]
        assert calc_rev == expected_rev
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_real_store_disaggregation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `real_store_disaggregation.py`**

Implement the constrained integer sampling algorithm using greedy coin-change with Dirichlet/multinomial item weighting according to student persona preferences (milk tea, fruit tea, toppings, iceblend on weekends, coffee on weekday mornings).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_real_store_disaggregation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fnb_forecast/data/real_store_disaggregation.py tests/data/test_real_store_disaggregation.py
git commit -m "feat(data): implement constrained integer disaggregation engine"
```

---

### Task 3: 14-Month Temporal Generator & 8-Table DataBundle Assembler

**Files:**
- Create: `src/fnb_forecast/data/real_store_generator.py`
- Test: `tests/data/test_real_store_generator.py`

**Interfaces:**
- Consumes: `parse_real_store_excel`, `disaggregate_revenue_series`, `get_real_store_item_master`, `get_real_store_recipes`, `get_real_store_ingredient_master`.
- Produces:
  - `generate_real_store_14m_bundle(excel_path: Path, seed: int = 42) -> DataBundle`
  - `save_real_store_bundle(bundle: DataBundle, output_dir: Path) -> None`

- [ ] **Step 1: Write failing test for 14-month bundle generation**

```python
# tests/data/test_real_store_generator.py
from pathlib import Path
from fnb_forecast.data.real_store_generator import generate_real_store_14m_bundle
from fnb_forecast.data.validation import validate_bundle

def test_generate_real_store_14m_bundle_validates():
    excel_path = Path("dataset.xlsx")
    if not excel_path.exists():
        excel_path = Path("f:/ugonnaloveit/dataset.xlsx")
    bundle = generate_real_store_14m_bundle(excel_path, seed=42)
    
    # Check date range: 2025-05-01 to 2026-06-30 (426 days)
    unique_dates = bundle.calendar["date"].unique()
    assert len(unique_dates) == 426
    
    # Validate with fnb_forecast validator
    validated = validate_bundle(bundle)
    assert validated is not None
    assert len(validated.daily_sales) == 43 * 426
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_real_store_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `real_store_generator.py`**

Build:
- Calendar generation with HCMC holidays, student calendar, and Tet 2026 store closure.
- Weather generator matching HCMC rain & temperature.
- Daily sales generation combining May & June real data with calibrated 12-month series.
- Inventory lots and scheduled receipts generation.
- Bundle saving with `provenance.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_real_store_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fnb_forecast/data/real_store_generator.py tests/data/test_real_store_generator.py
git commit -m "feat(data): assemble 14m calibrated real store DataBundle generator"
```

---

### Task 4: CLI Integration & End-to-End Pipeline Execution

**Files:**
- Modify: `src/fnb_forecast/cli.py`
- Test: `tests/integration/test_real_store_pipeline.py`

**Interfaces:**
- CLI command: `fnb-forecast generate-real-store --excel dataset.xlsx --output data/processed/real_store_14m`

- [ ] **Step 1: Write integration test for CLI generation and model training**

```python
# tests/integration/test_real_store_pipeline.py
from pathlib import Path
from typer.testing import CliRunner
from fnb_forecast.cli import app

runner = CliRunner()

def test_cli_generate_real_store(tmp_path: Path):
    excel_path = Path("dataset.xlsx")
    if not excel_path.exists():
        excel_path = Path("f:/ugonnaloveit/dataset.xlsx")
    output_dir = tmp_path / "real_store_14m"
    result = runner.invoke(app, [
        "generate-real-store",
        "--excel", str(excel_path),
        "--output", str(output_dir)
    ])
    assert result.exit_code == 0
    assert (output_dir / "daily_sales.csv").exists()
    assert (output_dir / "provenance.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_real_store_pipeline.py -v`
Expected: FAIL (command not registered)

- [ ] **Step 3: Modify `src/fnb_forecast/cli.py` to register `generate-real-store`**

Add the Typer command connecting `generate_real_store_14m_bundle` and `save_real_store_bundle`.

- [ ] **Step 4: Run test and generate dataset**

Run: `pytest tests/integration/test_real_store_pipeline.py -v`
Run command: `fnb-forecast generate-real-store --excel dataset.xlsx --output data/processed/real_store_14m`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fnb_forecast/cli.py tests/integration/test_real_store_pipeline.py
git commit -m "feat(cli): add generate-real-store CLI command and verify end-to-end"
```
