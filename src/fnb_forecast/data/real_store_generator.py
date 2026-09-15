"""14-month calibrated DataBundle generator for real-world Vietnam cafe store."""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from fnb_forecast.contracts import DataBundle, DataProvenance
from fnb_forecast.data.io import canonical_bundle_checksum, save_bundle
from fnb_forecast.data.real_store_disaggregation import (
    disaggregate_daily_revenue,
    get_category_priors,
)
from fnb_forecast.data.real_store_master import (
    get_real_store_ingredient_master,
    get_real_store_item_master,
    get_real_store_recipes,
    parse_real_store_excel,
)
from fnb_forecast.data.validation import validate_bundle


def generate_real_store_calendar(start_date: str = "2025-05-01", end_date: str = "2026-06-30") -> pd.DataFrame:
    """Generate 426-day calendar with Vietnam holidays, HCMC student cycles, and Tet 2026."""
    dates = pd.date_range(start_date, end_date, freq="D")
    records = []
    
    # Tet 2026: Mùng 1 Tết Bính Ngọ is 2026-02-17
    tet_2026 = pd.Timestamp("2026-02-17")
    
    # Vietnam public holidays
    holidays = {
        "2025-05-01": "Quốc tế Lao động",
        "2025-09-02": "Quốc khánh",
        "2025-12-24": "Giáng sinh",
        "2025-12-25": "Giáng sinh",
        "2026-01-01": "Tết Dương lịch",
        "2026-02-14": "Lễ Tình nhân (Valentine)",
        "2026-02-16": "29 Tết Bính Ngọ",
        "2026-02-17": "Mùng 1 Tết Bính Ngọ",
        "2026-02-18": "Mùng 2 Tết Bính Ngọ",
        "2026-02-19": "Mùng 3 Tết Bính Ngọ",
        "2026-04-26": "Giỗ tổ Hùng Vương (10/3 ÂL)",
        "2026-04-30": "Giải phóng Miền Nam",
        "2026-05-01": "Quốc tế Lao động",
    }
    
    # Closed dates: Store closed for Tet from Feb 15 to Feb 20, 2026
    # And closed on 2025-06-06 and 2025-06-08 (recorded 0 in Excel)
    closed_dates = {
        pd.Timestamp("2025-06-06"),
        pd.Timestamp("2025-06-08"),
        pd.Timestamp("2026-02-15"),
        pd.Timestamp("2026-02-16"),
        pd.Timestamp("2026-02-17"),
        pd.Timestamp("2026-02-18"),
        pd.Timestamp("2026-02-19"),
        pd.Timestamp("2026-02-20"),
    }
    
    for d in dates:
        d_norm = d.normalize()
        d_str = d.strftime("%Y-%m-%d")
        
        diff_tet = (d_norm - tet_2026).days
        days_to_tet = max(0, -diff_tet)
        days_after_tet = max(0, diff_tet)
        
        records.append({
            "date": d_norm,
            "weekday": d.weekday(),
            "is_weekend": d.weekday() >= 5,
            "holiday_name": holidays.get(d_str, None),
            "days_to_tet": days_to_tet,
            "days_after_tet": days_after_tet,
            "store_open": d_norm not in closed_dates,
        })
        
    return pd.DataFrame(records)


def generate_real_store_weather(calendar_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate realistic HCMC weather observations & 7-day ahead forecast issues."""
    rng = np.random.default_rng(seed)
    records = []
    
    dates = [pd.Timestamp(d).normalize() for d in calendar_df["date"]]
    end_date = max(dates)
    
    for d in dates:
        issue_date = d - pd.DateOffset(days=1)
        for h in range(0, 8):
            target_d = d + pd.DateOffset(days=h)
            if target_d > end_date:
                continue
                
            month = target_d.month
            is_rainy_season = 5 <= month <= 11
            
            if is_rainy_season:
                base_temp = 31.0 + rng.normal(0, 1.5)
                has_rain = rng.random() < 0.55
                rain = float(rng.gamma(shape=2.0, scale=8.0)) if has_rain else 0.0
                humidity = float(np.clip(75.0 + rng.normal(0, 6.0) + (10.0 if has_rain else 0.0), 50.0, 98.0))
            else:
                base_temp = 34.5 + rng.normal(0, 1.8)
                has_rain = rng.random() < 0.10
                rain = float(rng.gamma(shape=1.5, scale=4.0)) if has_rain else 0.0
                humidity = float(np.clip(60.0 + rng.normal(0, 5.0), 40.0, 85.0))
                
            temp = float(np.clip(base_temp, 24.0, 39.0))
            
            records.append({
                "issued_at": issue_date,
                "target_date": target_d,
                "temperature": round(temp, 1),
                "rain": round(rain, 1),
                "humidity": round(humidity, 1),
                "source": "previous_runs",
            })
        
    return pd.DataFrame(records)


def is_promo_item(item_id: str, category: str, date: pd.Timestamp) -> bool:
    """Determine if a menu item is on active promotion on a given date."""
    # Anchor period (May/June 2025) has no retroactively recorded promo flags
    if date.year == 2025 and date.month in [5, 6]:
        return False
        
    m = date.month
    d = date.day
    dow = date.weekday()
    
    # 1. Back-to-school campaign (01/09 - 10/09): Promo on Milktea & Fruit tea
    if date.year == 2025 and m == 9 and 1 <= d <= 10:
        if category in ["Milktea", "Tea_Soda"]:
            return True
            
    # 2. Halloween / Autumn Fest (25/10 - 31/10): Promo on Matcha & Cacao
    if date.year == 2025 and m == 10 and 25 <= d <= 31:
        if category in ["Matcha_Cacao", "Milktea"]:
            return True
            
    # 3. Black Friday / Cyber Week (24/11 - 30/11): All categories promo
    if date.year == 2025 and m == 11 and 24 <= d <= 30:
        if item_id in ["MT_TRUYENTHONG", "MT_DUONGDEN", "TEA_DAOCAMSA", "CF_MUOI", "ICE_MATCHA"]:
            return True
            
    # 4. Christmas & New Year Festival (20/12 - 02/01): Iceblend, Matcha, Special Coffee
    if (m == 12 and d >= 20) or (m == 1 and d <= 2):
        if category in ["Iceblend", "Matcha_Cacao"] or item_id in ["CF_TIRAMISU", "TOP_KEMTRUNG"]:
            return True
            
    # 5. Valentine & Women's Day (13/02 - 15/02 & 07/03 - 09/03): Sweet teas & Toppings
    if (m == 2 and 13 <= d <= 15) or (m == 3 and 7 <= d <= 9):
        if category in ["Topping", "Tea_Soda"] or "MAT" in item_id:
            return True
            
    # 6. Summer Heatwave Launch (20/04 - 05/05 & 15/06 - 25/06): Fruit teas & Iceblends
    if (m == 4 and d >= 20) or (m == 5 and d <= 5) or (m == 6 and 15 <= d <= 25):
        if category in ["Tea_Soda", "Iceblend"]:
            return True
            
    # 7. Happy Wednesday (Thứ Tư vui vẻ): Weekly discount on Milktea
    if dow == 2 and category == "Milktea":
        return True
        
    return False


def is_stockout_item(item_id: str, category: str, date: pd.Timestamp, rng: np.random.Generator) -> bool:
    """Simulate realistic out-of-stock events on high-demand or perishable items."""
    # Keep anchor months exact with no synthetic stockouts
    if date.year == 2025 and date.month in [5, 6]:
        return False
        
    dow = date.weekday()
    is_weekend = dow >= 4  # Fri, Sat, Sun
    
    # 1. Topping stockouts on busy weekend evenings
    if category == "Topping" and is_weekend:
        if rng.random() < 0.08:  # 8% chance per weekend day
            return True
            
    # 2. Perishable / Short shelf-life specialty drinks
    if item_id in ["MT_COMDEO", "MAT_GAU", "CAC_GAU", "TEA_MANGCAU"]:
        if rng.random() < 0.06:
            return True
            
    # 3. Iceblend stockout during peak heat or rush hours
    if category == "Iceblend" and is_weekend:
        if rng.random() < 0.05:
            return True
            
    # 4. General rare stockout across other items
    if rng.random() < 0.015:
        return True
        
    return False


def generate_real_store_daily_sales(
    excel_path: Path,
    calendar_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    items_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate 426-day daily sales for all 43 items:
    - May & June 2025: Exact real-world revenue from Excel disaggregated into items.
    - July 2025 - June 2026: Calibrated generative series with realistic promo and stockout dynamics.
    """
    df_rev_real, _ = parse_real_store_excel(excel_path)
    real_rev_lookup = {row["date"].normalize(): float(row["revenue"]) for _, row in df_rev_real.iterrows()}
    
    prices = {row["item_id"]: float(row["unit_price"]) for _, row in items_df.iterrows()}
    cat_lookup = {row["item_id"]: row["category"] for _, row in items_df.iterrows()}
    rng = np.random.default_rng(seed)
    
    sales_rows = []
    weather_lookup = {
        pd.Timestamp(row["target_date"]).normalize(): row
        for _, row in weather_df.drop_duplicates("target_date").iterrows()
    }
    
    for _, cal_row in calendar_df.iterrows():
        d = cal_row["date"].normalize()
        is_open = cal_row["store_open"]
        is_weekend = cal_row["is_weekend"]
        
        if not is_open:
            for item_id, price in prices.items():
                sales_rows.append({
                    "date": d,
                    "item_id": item_id,
                    "quantity": None,
                    "unit_price": price,
                    "promo_flag": False,
                    "stockout_flag": False,
                    "store_open": False,
                })
            continue
            
        # Check if in real anchor period (May 2025 or June 2025)
        if d.year == 2025 and d.month in [5, 6]:
            if d in real_rev_lookup:
                target_rev = real_rev_lookup[d]
            else:
                target_rev = 0.0
        else:
            month = d.month
            base_rev = 290000.0
            dow_mult = 1.35 if is_weekend else (1.15 if d.weekday() == 4 else 1.0)
            
            if month in [7, 8]:
                academic_mult = 0.85 if not is_weekend else 1.05
            elif month in [9, 10]:
                academic_mult = 1.28
            elif month in [11, 12]:
                academic_mult = 1.15
            elif month in [1, 2]:
                academic_mult = 1.20
            elif month in [3, 4, 5, 6]:
                academic_mult = 1.18
            else:
                academic_mult = 1.0
                
            w = weather_lookup.get(d)
            weather_mult = 1.0
            if w is not None:
                if w["rain"] > 25.0:
                    weather_mult *= 0.80
                if w["temperature"] > 35.0:
                    weather_mult *= 1.10
                    
            days_elapsed = (d - pd.Timestamp("2025-05-01")).days
            growth_mult = 1.0 + 0.10 * (days_elapsed / 426.0)
            noise = rng.normal(1.0, 0.08)
            
            calc_rev = base_rev * dow_mult * academic_mult * weather_mult * growth_mult * noise
            target_rev = float(max(50000, round(calc_rev / 1000.0) * 1000))
            
        # Prior weights with active promotional boosts
        priors = get_category_priors(items_df, is_weekend=is_weekend)
        for item_id in priors:
            if is_promo_item(item_id, cat_lookup[item_id], d):
                priors[item_id] *= 1.6  # Boost probability during promo
                
        day_seed = seed + int(d.timestamp()) % 100000
        quantities = disaggregate_daily_revenue(
            target_revenue=target_rev,
            item_prices=prices,
            priors=priors,
            is_weekend=is_weekend,
            seed=day_seed,
        )
        
        for item_id, price in prices.items():
            qty = quantities.get(item_id, 0)
            cat = cat_lookup[item_id]
            promo = is_promo_item(item_id, cat, d)
            stockout = is_stockout_item(item_id, cat, d, rng)
            
            if promo and qty > 0 and (d.year > 2025 or d.month > 6):
                # Extra promo boost
                qty += int(rng.integers(1, 3))
                
            if stockout and (d.year > 2025 or d.month > 6):
                # Truncate / reduce quantity due to stockout
                qty = max(0, int(round(qty * rng.uniform(0.1, 0.4))))
                
            sales_rows.append({
                "date": d,
                "item_id": item_id,
                "quantity": qty,
                "unit_price": price,
                "promo_flag": promo,
                "stockout_flag": stockout,
                "store_open": True,
            })
            
    df_sales = pd.DataFrame(sales_rows).sort_values(["item_id", "date"]).reset_index(drop=True)
    return df_sales


def generate_real_store_inventory(
    ingredients_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate starting inventory lots and scheduled inbound receipts."""
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp(calendar_df["date"].min()).normalize()
    
    lots = []
    receipts = []
    
    for idx, row in ingredients_df.iterrows():
        ing_id = row["ingredient_id"]
        pack_size = row["pack_size"]
        shelf_life = int(row["shelf_life_days"])
        
        # Initial lot
        days_back = int(rng.integers(1, min(4, shelf_life + 1)))
        received_date = start_date - pd.DateOffset(days=days_back)
        expiry_date = received_date + pd.DateOffset(days=shelf_life)
        on_hand = float(pack_size * rng.uniform(2.0, 5.0))
        
        lots.append({
            "ingredient_id": ing_id,
            "lot_id": f"LOT_{ing_id}_01",
            "on_hand": round(on_hand, 2),
            "expiry_date": pd.Timestamp(expiry_date).normalize(),
            "received_date": pd.Timestamp(received_date).normalize(),
        })
        
        # Scheduled receipts in early days
        if shelf_life <= 7:
            arrival_date = start_date + pd.DateOffset(days=2)
            receipts.append({
                "ingredient_id": ing_id,
                "arrival_date": pd.Timestamp(arrival_date).normalize(),
                "quantity": float(pack_size * 3.0),
            })
            
    df_lots = pd.DataFrame(lots)
    df_receipts = pd.DataFrame(receipts) if receipts else pd.DataFrame(columns=["ingredient_id", "arrival_date", "quantity"])
    
    return df_lots, df_receipts


def generate_real_store_14m_bundle(excel_path: Path, seed: int = 42) -> DataBundle:
    """Build and return a complete, schema-validated 14-month DataBundle for the real cafe store."""
    items_df = get_real_store_item_master()
    ingredients_df = get_real_store_ingredient_master()
    recipes_df = get_real_store_recipes()
    
    calendar_df = generate_real_store_calendar(start_date="2025-05-01", end_date="2026-06-30")
    weather_df = generate_real_store_weather(calendar_df, seed=seed)
    sales_df = generate_real_store_daily_sales(excel_path, calendar_df, weather_df, items_df, seed=seed)
    lots_df, receipts_df = generate_real_store_inventory(ingredients_df, calendar_df, seed=seed)
    
    # Trim items_df columns to required schema
    items_clean = items_df[["item_id", "item_name", "category", "launch_date", "active"]].copy()
    
    # Trim ingredient_master to schema
    ing_clean = ingredients_df[["ingredient_id", "name", "base_unit", "pack_size", "lead_time_days", "review_period_days", "shelf_life_days"]].copy()
    
    provenance = DataProvenance(
        name="real_store_14m",
        source="dataset.xlsx + calibrated_disaggregation",
        is_synthetic=False,
        generated_at=datetime.now(),
        checksum=None,
    )
    
    bundle = DataBundle(
        daily_sales=sales_df,
        item_master=items_clean,
        calendar=calendar_df,
        weather=weather_df,
        recipes=recipes_df,
        ingredient_master=ing_clean,
        inventory_lots=lots_df,
        scheduled_receipts=receipts_df,
        provenance=provenance,
    )
    
    # Validate bundle
    validated = validate_bundle(bundle)
    
    # Add checksum to provenance
    checksum = canonical_bundle_checksum(validated)
    validated.provenance = DataProvenance(
        name=provenance.name,
        source=provenance.source,
        is_synthetic=provenance.is_synthetic,
        generated_at=provenance.generated_at,
        checksum=checksum,
    )
    
    return validated


def save_real_store_bundle(bundle: DataBundle, output_dir: Path) -> None:
    """Save the validated DataBundle to the target directory."""
    save_bundle(bundle, output_dir)
