"""Ingredient demand and order proposal page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fnb_forecast.app.components import (
    inject_custom_css,
    render_download,
    render_provenance,
    render_warnings,
    table_or_message,
)
from fnb_forecast.app.components import render_domain_error
from fnb_forecast.app.state import DATASET_OPTIONS, DEFAULT_DATASET, get_bundle, get_latest_output, run_forecast
from fnb_forecast.exceptions import ForecastingDomainError

st.set_page_config(page_title="Kế hoạch nguyên liệu & đặt hàng · F&B Forecast", page_icon=":material/inventory_2:", layout="wide")

inject_custom_css()
output = get_latest_output()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### :material/tune: Đặt hàng & tồn kho")
    with st.container(border=True):
        current_sl = float(output.metadata.get("service_level", 0.95))
        sl_index = 0 if current_sl == 0.95 else (1 if current_sl == 0.98 else 2)
        
        service_level_map = {
            0.95: "95%",
            0.98: "98%",
            0.90: "90%",
        }
        sl_label = st.selectbox(
            "Mức độ đảm bảo hàng kho (Service Level)",
            list(service_level_map.values()),
            index=sl_index,
            help="Độ tin cậy để tính toán lượng tồn kho an toàn và đề xuất đơn đặt hàng."
        )
        selected_sl = [k for k, v in service_level_map.items() if v == sl_label][0]
        
        if st.button("Cập nhật kế hoạch đặt hàng", type="primary", width="stretch", icon=":material/refresh:"):
            try:
                origin_str = output.metadata.get("forecast_origin", "2026-06-23")
                origin_dt = pd.Timestamp(str(origin_str).split(" ")[0])
                current_ds = str(output.metadata.get("provenance", output.metadata.get("dataset_name", DEFAULT_DATASET)))
                dataset_key = st.session_state.get("selected_dataset", DEFAULT_DATASET)
                for key in DATASET_OPTIONS:
                    if key in current_ds:
                        dataset_key = key
                        break
                selected_model = st.session_state.get("selected_model_id", output.metadata.get("requested_model", "auto"))
                with st.spinner("Đang tính toán lại lượng tồn kho an toàn và đề xuất FEFO..."):
                    output = run_forecast(origin_dt, float(selected_sl), dataset=dataset_key, model_id=selected_model)
                st.toast("Đã cập nhật kế hoạch đặt hàng mới!", icon=":material/check_circle:")
                st.rerun()
            except (ForecastingDomainError, OSError, ValueError, TypeError) as error:
                render_domain_error(error)

# --- Header ---
header_col1, header_col2 = st.columns([3, 1], vertical_alignment="center")
with header_col1:
    st.title(":material/inventory_2: 2 · Kế hoạch nguyên liệu & đặt hàng")
    st.caption("Tính toán chính xác lượng nguyên vật liệu cần dùng và tạo đề xuất đặt hàng tự động")
with header_col2:
    render_provenance(output)

# --- Quick Stats KPI ---
ing_count = output.ingredient_forecast["ingredient_id"].nunique() if not output.ingredient_forecast.empty and "ingredient_id" in output.ingredient_forecast else len(output.ingredient_forecast)
orders = output.order_proposals
needed_orders_count = int((orders["suggested_order"].fillna(0) > 0).sum()) if not orders.empty and "suggested_order" in orders else 0
waste_alerts = int(orders["waste_risk_warning"].fillna(False).astype(bool).sum()) if not orders.empty and "waste_risk_warning" in orders else 0

with st.container(horizontal=True):
    st.metric("Tổng nguyên liệu cần dùng", f"{ing_count} loại", border=True)
    st.metric("Mặt hàng cần gửi đơn đặt", f"{needed_orders_count} loại", delta="Cần tạo đơn" if needed_orders_count > 0 else "Đã đủ", delta_color="off" if needed_orders_count > 0 else "normal", border=True)
    st.metric("Cảnh báo rủi ro cận date/hết hạn", f"{waste_alerts} loại", delta="Cần lưu ý" if waste_alerts > 0 else "An toàn", delta_color="inverse" if waste_alerts > 0 else "normal", border=True)

st.write("")

# --- Demand & Proposals Tabs ---
with st.container(border=True):
    tab_orders, tab_demand, tab_inventory, tab_downloads = st.tabs([
        ":material/local_shipping: Đề xuất đặt hàng ngay (Order proposals)",
        ":material/soup_kitchen: Nhu cầu tiêu thụ theo ngày (Daily needs)",
        ":material/inventory: Tồn kho hiện có theo lô (Inventory lots)",
        ":material/download: Tải file gửi nhà cung cấp",
    ])
    
    with tab_orders:
        st.subheader("Danh sách nguyên liệu đề xuất đặt mua")
        st.caption("Số lượng được tính toán dựa trên mức tồn kho an toàn và lượng tiêu thụ dự kiến 7 ngày tới, đã làm tròn theo quy cách đóng gói.")
        
        if not orders.empty:
            table_or_message(orders)
        else:
            st.info("Chưa có đề xuất đặt hàng nào.")
        render_warnings(output.warnings)
        
    with tab_demand:
        st.subheader("Nhu cầu nguyên liệu chi tiết theo từng ngày")
        st.caption("Giúp bếp và quầy pha chế chủ động kế hoạch sơ chế nguyên liệu hàng ngày.")
        table_or_message(output.ingredient_forecast)

    with tab_inventory:
        st.subheader("Tồn kho hiện có theo lô (bao gồm hạn sử dụng)")
        bundle = get_bundle()
        if bundle is not None and not bundle.inventory_lots.empty:
            lots_display = bundle.inventory_lots.copy()
            if not bundle.ingredient_master.empty:
                ing_meta = bundle.ingredient_master.copy()
                if "name" in ing_meta.columns and "ingredient_name" not in ing_meta.columns:
                    ing_meta = ing_meta.rename(columns={"name": "ingredient_name"})
                cols_to_use = [c for c in ["ingredient_id", "ingredient_name", "base_unit"] if c in ing_meta.columns]
                lots_display = lots_display.merge(ing_meta[cols_to_use], on="ingredient_id", how="left")
                if "ingredient_name" in lots_display.columns:
                    lots_display["ingredient_name"] = lots_display["ingredient_name"].fillna(lots_display["ingredient_id"])

            table_or_message(lots_display)
            st.caption("Dữ liệu thể hiện số lượng thực tế trong kho và ngày hết hạn của từng lô hàng.")
        else:
            st.info("Chưa có dữ liệu tồn kho theo lô.")
        
    with tab_downloads:
        st.markdown("**Xuất danh sách đặt hàng để gửi nhà cung cấp hoặc lưu trữ nội bộ:**")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            render_download(
                "Tải file đề xuất đặt hàng (CSV)",
                output.order_proposals,
                "phieu_de_xuat_dat_hang.csv",
                output.warnings,
            )
        with d_col2:
            render_download(
                "Tải file nhu cầu nguyên liệu chi tiết (CSV)",
                output.ingredient_forecast,
                "nhu_cau_nguyen_lieu_7ngay.csv",
            )

