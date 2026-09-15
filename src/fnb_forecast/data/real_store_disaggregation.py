"""Constrained integer disaggregation engine matching daily store revenue to menu items."""

from typing import Dict, List
import numpy as np
import pandas as pd

from fnb_forecast.data.real_store_master import get_real_store_item_master


def get_category_priors(items_df: pd.DataFrame, is_weekend: bool = False) -> Dict[str, float]:
    """
    Return prior preference weights for menu items reflecting youth / student persona in HCMC.
    Weekends favor Iceblend, premium milk teas, matcha, and toppings.
    Weekdays favor classic coffees and fruit teas.
    """
    priors = {}
    for _, row in items_df.iterrows():
        item_id = row["item_id"]
        cat = row["category"]
        
        if cat == "Milktea":
            base = 2.5
            if item_id in ["MT_TRUYENTHONG", "MT_DUONGDEN", "MT_THAI"]:
                base = 3.5
        elif cat == "Tea_Soda":
            base = 2.2
            if item_id in ["TEA_DAOCAMSA", "TEA_DAU", "TEA_OIHONG", "TEA_MANGCAU"]:
                base = 3.0
        elif cat == "Coffee":
            base = 2.0 if not is_weekend else 1.2
            if item_id in ["CF_BACXIU", "CF_MUOI", "CF_SUA"]:
                base += 0.8
        elif cat == "Iceblend":
            base = 1.2 if not is_weekend else 2.8
        elif cat == "Matcha_Cacao":
            base = 1.5 if not is_weekend else 2.2
        elif cat == "Topping":
            base = 1.8  # Attached to drinks
        elif cat == "Dessert":
            base = 0.8
        else:
            base = 1.0
            
        priors[item_id] = base
        
    return priors


def _get_reachability_table(max_thousands: int = 1500, prices_in_k: tuple[int, ...] = (5, 7, 10, 15, 17, 20, 25, 30)) -> list[bool]:
    """Precompute reachability array in thousands (k VND)."""
    dp = [False] * (max_thousands + 1)
    dp[0] = True
    for p in sorted(prices_in_k):
        for v in range(p, max_thousands + 1):
            if dp[v - p]:
                dp[v] = True
    return dp


_GLOBAL_REACHABILITY_DP = _get_reachability_table()


def disaggregate_daily_revenue(
    target_revenue: float,
    item_prices: Dict[str, float],
    priors: Dict[str, float],
    is_weekend: bool = False,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Disaggregate a single day's total revenue into integer item quantities
    such that sum(quantity * unit_price) == target_revenue.
    """
    target = int(round(target_revenue))
    if target <= 0:
        return {item_id: 0 for item_id in item_prices}
        
    rng = np.random.default_rng(seed)
    
    # Scale all prices to integer in thousands (k VND)
    int_prices_k = {k: int(round(v / 1000.0)) for k, v in item_prices.items()}
    target_k = int(round(target / 1000.0))
    
    dp = _GLOBAL_REACHABILITY_DP
    item_ids = list(item_prices.keys())
    quantities = {item_id: 0 for item_id in item_ids}
    
    rem_k = target_k
    while rem_k > 0:
        valid_items = []
        valid_weights = []
        
        for item_id in item_ids:
            pk = int_prices_k[item_id]
            if pk <= rem_k and (rem_k - pk < len(dp)) and dp[rem_k - pk]:
                valid_items.append(item_id)
                valid_weights.append(priors.get(item_id, 1.0))
                
        if not valid_items:
            break
            
        probs = np.array(valid_weights, dtype=float)
        probs /= probs.sum()
        
        chosen_item = rng.choice(valid_items, p=probs)
        quantities[chosen_item] += 1
        rem_k -= int_prices_k[chosen_item]
        
    return quantities


def disaggregate_revenue_series(
    daily_rev_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Disaggregate a daily revenue DataFrame (with columns ['date', 'revenue'])
    into standard daily_sales format across all 43 menu items.
    """
    items_df = get_real_store_item_master()
    prices = {row["item_id"]: float(row["unit_price"]) for _, row in items_df.iterrows()}
    
    rows = []
    
    for idx, row in daily_rev_df.iterrows():
        date = pd.Timestamp(row["date"]).normalize()
        revenue = float(row["revenue"])
        is_weekend = date.weekday() >= 5
        
        priors = get_category_priors(items_df, is_weekend=is_weekend)
        day_seed = seed + int(date.timestamp()) % 100000
        
        quantities = disaggregate_daily_revenue(
            target_revenue=revenue,
            item_prices=prices,
            priors=priors,
            is_weekend=is_weekend,
            seed=day_seed,
        )
        
        for item_id, price in prices.items():
            qty = quantities.get(item_id, 0)
            rows.append({
                "date": date,
                "item_id": item_id,
                "quantity": qty,
                "unit_price": price,
                "promo_flag": False,
                "stockout_flag": False,
                "store_open": True,
            })
            
    result = pd.DataFrame(rows).sort_values(["item_id", "date"]).reset_index(drop=True)
    return result
