"""Day-by-day forecast and moving-average inventory policy simulation."""

from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SimulationResult:
    policy_summary: pd.DataFrame
    daily_audit: pd.DataFrame
    metadata: dict[str, Any]


class InventorySimulator:
    """Run two policies from identical input state and satisfy demand with FEFO."""

    def __init__(
        self,
        *,
        actual_need: pd.DataFrame,
        model_forecasts: pd.DataFrame,
        inventory_lots: pd.DataFrame,
        scheduled_receipts: pd.DataFrame,
        ingredient_master: pd.DataFrame,
    ) -> None:
        self.actual_need = actual_need.copy()
        self.model_forecasts = model_forecasts.copy()
        self.inventory_lots = inventory_lots.copy()
        self.scheduled_receipts = scheduled_receipts.copy()
        self.ingredient_master = ingredient_master.copy()

    def run(self) -> SimulationResult:
        audits: list[pd.DataFrame] = []
        for policy in ("forecast", "baseline"):
            audits.append(self._run_policy_real(policy))
        daily = pd.concat(audits, ignore_index=True)
        summary_rows: list[dict[str, object]] = []
        for policy, rows in daily.groupby("policy", sort=True):
            total_need = float(rows["actual_need"].sum())
            summary_rows.append(
                {
                    "policy": policy,
                    "fill_rate": float(rows["served"].sum() / total_need) if total_need else 1.0,
                    "stockout_days": int(
                        rows["unmet"].gt(0).groupby(rows["target_date"]).any().sum()
                    ),
                    "waste_rate": float(rows["waste"].sum() / rows["received"].sum())
                    if rows["received"].sum()
                    else 0.0,
                    "average_inventory": float(rows["ending_inventory"].mean()),
                    "total_ordered": float(rows["ordered"].sum()),
                }
            )
        return SimulationResult(
            policy_summary=pd.DataFrame(summary_rows),
            daily_audit=daily,
            metadata={"policies": ["forecast", "baseline"], "order_timing": "before_demand"},
        )

    def _run_policy(self, policy: str) -> pd.DataFrame:
        lots = self.inventory_lots.copy()
        rows: list[dict[str, object]] = []
        dates = (
            pd.to_datetime(self.actual_need["target_date"], errors="raise")
            .dt.normalize()
            .sort_values()
            .unique()
        )
        for date in dates:
            day = self.actual_need.loc[
                pd.to_datetime(self.actual_need["target_date"]).dt.normalize().eq(date)
            ]
            for _, demand in day.iterrows():
                ingredient_id = demand["ingredient_id"]
                amount = float(demand["actual_need"])
                usable = lots.loc[
                    lots["ingredient_id"].eq(ingredient_id)
                    & pd.to_datetime(lots["expiry_date"]).dt.normalize().ge(pd.Timestamp(date))
                ]
                available = float(usable["on_hand"].sum())
                served = min(amount, available)
                remaining = served
                for index in usable.sort_values(["expiry_date", "received_date", "lot_id"]).index:
                    used = min(float(lots.loc[index, "on_hand"]), remaining)
                    lots.loc[index, "on_hand"] -= used
                    remaining -= used
                    if remaining <= 0:
                        break
                rows.append(
                    {
                        "policy": policy,
                        "target_date": date,
                        "ingredient_id": ingredient_id,
                        "actual_need": amount,
                        "served": served,
                        "unmet": amount - served,
                        "received": 0.0,
                        "ordered": 0.0,
                        "waste": 0.0,
                        "ending_inventory": float(lots["on_hand"].sum()),
                    }
                )
    def _run_policy_real(self, policy: str) -> pd.DataFrame:
        """Simulate receipts, ordering, expiry, and FEFO consumption day by day."""
        lots = self.inventory_lots.copy()
        lots["expiry_date"] = pd.to_datetime(lots["expiry_date"]).dt.normalize()
        lots["received_date"] = pd.to_datetime(lots["received_date"]).dt.normalize()
        receipts = self.scheduled_receipts.copy()
        if not receipts.empty:
            receipts["arrival_date"] = pd.to_datetime(receipts["arrival_date"]).dt.normalize()
        forecast = self.model_forecasts.copy()
        if not forecast.empty:
            forecast["target_date"] = pd.to_datetime(forecast["target_date"]).dt.normalize()
        actual = self.actual_need.copy()
        actual["target_date"] = pd.to_datetime(actual["target_date"]).dt.normalize()
        history: dict[object, list[float]] = {}
        pending: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        for raw_date in sorted(actual["target_date"].unique()):
            date = pd.Timestamp(raw_date)
            received_today = 0.0
            due = [*receipts.to_dict("records"), *pending]
            pending = [r for r in pending if pd.Timestamp(r["arrival_date"]).normalize() > date]
            for receipt in due:
                if pd.Timestamp(receipt["arrival_date"]).normalize() != date:
                    continue
                ingredient_id = receipt["ingredient_id"]
                master = self.ingredient_master.loc[
                    self.ingredient_master["ingredient_id"].eq(ingredient_id)
                ].iloc[0]
                shelf_life = master.get("shelf_life_days")
                expiry = date + pd.Timedelta(int(shelf_life), unit="D") if pd.notna(shelf_life) else date + pd.Timedelta(3650, unit="D")
                lots.loc[len(lots)] = {
                    "ingredient_id": ingredient_id,
                    "lot_id": receipt.get("lot_id", f"receipt-{len(lots)+1}"),
                    "on_hand": float(receipt["quantity"]),
                    "expiry_date": expiry,
                    "received_date": date,
                }
                received_today += float(receipt["quantity"])
            expired = lots["expiry_date"].lt(date) & lots["on_hand"].gt(0)
            waste_today = float(lots.loc[expired, "on_hand"].sum())
            lots.loc[expired, "on_hand"] = 0.0
            for ingredient_id, day in actual.loc[actual["target_date"].eq(date)].groupby("ingredient_id"):
                master = self.ingredient_master.loc[
                    self.ingredient_master["ingredient_id"].eq(ingredient_id)
                ].iloc[0]
                lead = int(master["lead_time_days"])
                review = int(master["review_period_days"])
                coverage_end = date + pd.Timedelta(lead + review - 1, unit="D")
                if policy == "forecast":
                    target_need = float(forecast.loc[
                        forecast["ingredient_id"].eq(ingredient_id)
                        & forecast["target_date"].between(date, coverage_end),
                        "forecast_need",
                    ].sum())
                else:
                    past = history.get(ingredient_id, [])[-7:]
                    average = sum(past) / len(past) if past else float(day["actual_need"].mean())
                    target_need = average * (lead + review)
                usable = lots.loc[
                    lots["ingredient_id"].eq(ingredient_id)
                    & lots["expiry_date"].ge(coverage_end)
                    & lots["on_hand"].gt(0)
                ]
                scheduled = sum(
                    float(r["quantity"])
                    for r in [*receipts.to_dict("records"), *pending]
                    if r["ingredient_id"] == ingredient_id
                    and pd.Timestamp(r["arrival_date"]).normalize() <= coverage_end
                )
                pack = float(master["pack_size"])
                raw_order = max(0.0, target_need - float(usable["on_hand"].sum()) - scheduled)
                ordered = math.ceil(raw_order / pack) * pack if raw_order > 0 else 0.0
                if ordered > 0:
                    pending.append({
                        "ingredient_id": ingredient_id,
                        "arrival_date": date + pd.Timedelta(lead, unit="D"),
                        "quantity": ordered,
                        "lot_id": f"{policy}-{ingredient_id}-{date.date()}",
                    })
                amount = float(day["actual_need"].sum())
                usable_today = lots.loc[
                    lots["ingredient_id"].eq(ingredient_id)
                    & lots["expiry_date"].ge(date)
                    & lots["on_hand"].gt(0)
                ].sort_values(["expiry_date", "received_date", "lot_id"])
                served = min(amount, float(usable_today["on_hand"].sum()))
                remaining = served
                for index in usable_today.index:
                    used = min(float(lots.loc[index, "on_hand"]), remaining)
                    lots.loc[index, "on_hand"] -= used
                    remaining -= used
                    if remaining <= 0:
                        break
                rows.append({
                    "policy": policy, "target_date": date, "ingredient_id": ingredient_id,
                    "actual_need": amount, "served": served, "unmet": amount - served,
                    "received": received_today, "ordered": ordered, "waste": waste_today,
                    "ending_inventory": float(lots["on_hand"].sum()),
                })
                history.setdefault(ingredient_id, []).append(amount)
        return pd.DataFrame(rows)
        return pd.DataFrame(rows)
