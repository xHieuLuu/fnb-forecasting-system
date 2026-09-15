"""Item-level forecast page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fnb_forecast.app.components import (
    inject_custom_css,
    render_download,
    render_interval_chart,
    render_provenance,
    table_or_message,
    vnd,
)
from fnb_forecast.app.state import get_bundle, get_latest_output

st.set_page_config(page_title="Dự báo món bán ra · F&B Forecast", page_icon=":material/local_cafe:", layout="wide")

inject_custom_css()
output = get_latest_output()

# --- Header ---
header_col1, header_col2 = st.columns([3, 1], vertical_alignment="center")
with header_col1:
    st.title(":material/local_cafe: 1 · Dự báo số lượng món bán ra")
    st.caption("Dự báo nhu cầu tiêu thụ chi tiết từng món theo ngày trong 7 ngày tới")
with header_col2:
    render_provenance(output)

frame = output.item_forecasts.copy()
rev_frame = output.revenue_forecast.copy()

if frame.empty:
    table_or_message(frame, "Chưa có dữ liệu dự báo món. Vui lòng quay lại trang chủ và bấm chạy dự báo.")
    st.stop()

# Merge revenue if available
if not rev_frame.empty and {"target_date", "item_id", "forecast_revenue"}.issubset(rev_frame.columns):
    frame = frame.merge(
        rev_frame[["target_date", "item_id", "forecast_revenue"]],
        on=["target_date", "item_id"],
        how="left",
    )

# Try merging actual sales & item metadata if available in bundle
bundle = get_bundle()
if bundle is not None and not frame.empty:
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if hasattr(bundle, "daily_sales") and not bundle.daily_sales.empty:
        actual = bundle.daily_sales.rename(columns={"date": "target_date", "quantity": "actual"})
        actual = actual.loc[:, ["target_date", "item_id", "actual"]]
        actual["target_date"] = pd.to_datetime(actual["target_date"], errors="coerce")
        frame = frame.merge(actual, on=["target_date", "item_id"], how="left")
    
    if not bundle.item_master.empty:
        cols = [c for c in ["item_id", "item_name", "category"] if c in bundle.item_master.columns]
        item_meta = bundle.item_master.loc[:, cols].drop_duplicates()
        missing_cols = [c for c in ["item_name", "category"] if c in item_meta.columns and c not in frame.columns]
        if missing_cols:
            frame = frame.merge(item_meta[["item_id"] + missing_cols], on="item_id", how="left")

# --- Filters Container ---
with st.container(border=True):
    st.markdown("##### :material/filter_alt: Bộ lọc danh mục & món")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        if "category" in frame and frame["category"].dropna().astype(str).nunique() > 0:
            all_categories = sorted(frame["category"].dropna().astype(str).unique())
            selected_categories = st.multiselect("Lọc theo danh mục", all_categories, default=all_categories)
            if selected_categories:
                frame = frame.loc[frame["category"].astype(str).isin(selected_categories)].copy()
        else:
            all_categories = ["Tất cả danh mục"]
            st.multiselect("Lọc theo danh mục", all_categories, default=all_categories, disabled=True)

    with f_col2:
        item_col = "item_name" if "item_name" in frame else "item_id"
        available_items = sorted(frame[item_col].dropna().astype(str).unique())
        selected_items = st.multiselect(
            "Lọc theo tên món",
            available_items,
            default=available_items,
        )
        if selected_items:
            frame = frame.loc[frame[item_col].astype(str).isin(selected_items)].copy()

st.write("")

# --- KPI summary of filtered items ---
selected_cups = float(frame["yhat"].sum()) if "yhat" in frame else 0.0
selected_rev = float(frame["forecast_revenue"].sum()) if "forecast_revenue" in frame else 0.0
item_count = frame[item_col].nunique() if item_col in frame else 0

with st.container(horizontal=True):
    st.metric("Số món đang lọc", f"{item_count} món", border=True)
    st.metric("Tổng lượng dự kiến bán", f"{selected_cups:,.0f} ly", border=True)
    st.metric("Doanh thu ước tính", vnd(selected_rev), border=True)

st.write("")

# --- Chart Card ---
with st.container(border=True):
    st.subheader(":material/area_chart: Xu hướng nhu cầu theo ngày")
    st.caption("Tổng hợp số lượng dự kiến bán ra mỗi ngày của các món đang chọn (kèm khoảng dao động an toàn)")
    
    # Aggregate by date for the chart
    if not frame.empty and "target_date" in frame:
        agg_chart = frame.groupby("target_date", as_index=False).agg({
            c: "sum" for c in ["yhat", "lower", "upper", "actual"] if c in frame.columns
        })
        render_interval_chart(agg_chart, value_column="yhat")
    else:
        st.info("Chưa có dữ liệu biểu đồ.")

st.write("")

# --- Table Card ---
with st.container(border=True):
    t_head1, t_head2 = st.columns([3, 1], vertical_alignment="center")
    with t_head1:
        st.subheader(":material/table_chart: Bảng chi tiết dự báo từng món")
    with t_head2:
        render_download(
            "Tải file CSV đã lọc",
            frame,
            "du_bao_mon_chi_tiet.csv",
            output.warnings,
        )
    
    table_or_message(frame, "Không có dòng phù hợp với bộ lọc đã chọn.")

