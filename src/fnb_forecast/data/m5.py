"""Load a leakage-safe subset of the public M5 benchmark."""

from pathlib import Path

import pandas as pd

from fnb_forecast.contracts import DataBundle, DataProvenance

from .io import canonical_bundle_checksum
from .validation import validate_bundle

M5_SOURCE = "https://www.kaggle.com/competitions/m5-forecasting-accuracy"


def select_food_items(
    sales: pd.DataFrame, selection_end: pd.Timestamp, n_items: int
) -> list[str]:
    """Select food items using demand observed no later than ``selection_end``."""
    eligible = sales.loc[
        sales["date"].le(selection_end) & sales["category"].eq("FOODS")
    ]
    ranked = (
        eligible.groupby("item_id", as_index=False)["quantity"]
        .agg(nonzero_days=lambda values: values.gt(0).sum(), total_units="sum")
        .sort_values(["nonzero_days", "total_units", "item_id"], ascending=[False, False, True])
    )
    return ranked.head(n_items)["item_id"].tolist()


def _empty_frame(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


def _m5_calendar(calendar: pd.DataFrame, observed_days: pd.Series) -> pd.DataFrame:
    frame = calendar.loc[calendar["d"].isin(observed_days), ["date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    frame["weekday"] = frame["date"].dt.weekday
    frame["is_weekend"] = frame["weekday"].ge(5).astype("boolean")
    frame["holiday_name"] = pd.NA
    frame["days_to_tet"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["days_after_tet"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["store_open"] = pd.Series(True, index=frame.index, dtype="boolean")
    return frame


def load_m5(
    raw_dir: Path, store_id: str, n_items: int, selection_end: pd.Timestamp
) -> DataBundle:
    """Normalize one store's selected M5 FOODS demand into the shared data contract."""
    raw_dir = Path(raw_dir)
    calendar_raw = pd.read_csv(raw_dir / "calendar.csv", encoding="utf-8")
    prices = pd.read_csv(raw_dir / "sell_prices.csv", encoding="utf-8")
    wide_sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv", encoding="utf-8")
    day_columns = [column for column in wide_sales.columns if column.startswith("d_")]
    id_columns = ["item_id", "cat_id", "store_id"]
    long_sales = wide_sales.loc[
        wide_sales["store_id"].eq(store_id) & wide_sales["cat_id"].eq("FOODS"),
        id_columns + day_columns,
    ].melt(id_vars=id_columns, var_name="d", value_name="quantity")
    long_sales = long_sales.merge(
        calendar_raw.loc[:, ["d", "date", "wm_yr_wk"]], on="d", how="inner", validate="many_to_one"
    )
    long_sales["date"] = pd.to_datetime(long_sales["date"], errors="raise")
    long_sales["quantity"] = pd.to_numeric(long_sales["quantity"], errors="raise")
    long_sales = long_sales.rename(columns={"cat_id": "category"})
    selected_items = select_food_items(long_sales, pd.Timestamp(selection_end), n_items)
    if len(selected_items) != n_items:
        raise ValueError(f"requested {n_items} FOODS items but only found {len(selected_items)}")
    selected = long_sales.loc[long_sales["item_id"].isin(selected_items)].copy()
    selected = selected.merge(
        prices.loc[:, ["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    if selected["sell_price"].isna().any():
        raise ValueError("M5 sell_prices.csv is missing a price for selected store-item-week rows")

    daily_sales = selected.loc[
        :, ["date", "item_id", "quantity", "sell_price"]
    ].rename(columns={"sell_price": "unit_price"})
    for column in ("promo_flag", "stockout_flag"):
        daily_sales[column] = pd.Series(False, index=daily_sales.index, dtype="boolean")
    daily_sales["store_open"] = pd.Series(True, index=daily_sales.index, dtype="boolean")
    launch_dates = selected.groupby("item_id")["date"].min().reindex(selected_items).tolist()
    item_master = pd.DataFrame(
        {
            "item_id": selected_items,
            "item_name": selected_items,
            "category": "FOODS",
            "launch_date": launch_dates,
            "active": pd.Series(True, index=range(len(selected_items)), dtype="boolean"),
        }
    )
    bundle = validate_bundle(
        DataBundle(
            daily_sales=daily_sales,
            item_master=item_master,
            calendar=_m5_calendar(calendar_raw, selected["d"]),
            weather=_empty_frame(
                {
                    "issued_at": "datetime64[ns]",
                    "target_date": "datetime64[ns]",
                    "temperature": "float64",
                    "rain": "float64",
                    "humidity": "float64",
                    "source": "object",
                }
            ),
            recipes=_empty_frame(
                {
                    "item_id": "object",
                    "ingredient_id": "object",
                    "amount": "float64",
                    "unit": "object",
                    "waste_rate": "float64",
                }
            ),
            ingredient_master=_empty_frame(
                {
                    "ingredient_id": "object",
                    "name": "object",
                    "base_unit": "object",
                    "pack_size": "float64",
                    "lead_time_days": "float64",
                    "review_period_days": "float64",
                    "shelf_life_days": "float64",
                }
            ),
            inventory_lots=_empty_frame(
                {
                    "ingredient_id": "object",
                    "lot_id": "object",
                    "on_hand": "float64",
                    "expiry_date": "datetime64[ns]",
                    "received_date": "datetime64[ns]",
                }
            ),
            scheduled_receipts=_empty_frame(
                {
                    "ingredient_id": "object",
                    "arrival_date": "datetime64[ns]",
                    "quantity": "float64",
                }
            ),
            provenance=DataProvenance(
                name="m5_public_benchmark",
                source=M5_SOURCE,
                is_synthetic=False,
                generated_at=None,
                checksum=None,
            ),
        )
    )
    return DataBundle(
        daily_sales=bundle.daily_sales,
        item_master=bundle.item_master,
        calendar=bundle.calendar,
        weather=bundle.weather,
        recipes=bundle.recipes,
        ingredient_master=bundle.ingredient_master,
        inventory_lots=bundle.inventory_lots,
        scheduled_receipts=bundle.scheduled_receipts,
        provenance=DataProvenance(
            name="m5_public_benchmark",
            source=M5_SOURCE,
            is_synthetic=False,
            generated_at=None,
            checksum=canonical_bundle_checksum(bundle),
        ),
    )
