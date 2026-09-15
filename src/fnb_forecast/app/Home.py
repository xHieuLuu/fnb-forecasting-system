"""Dashboard overview and forecast runner."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from fnb_forecast.app.components import (
    friendly_dataset_name,
    inject_custom_css,
    item_metrics,
    render_domain_error,
    render_download,
    render_metadata_download,
    render_provenance,
    render_warnings,
    table_or_message,
    top_warnings,
    vnd,
)
from fnb_forecast.app.state import DATASET_OPTIONS, DEFAULT_DATASET, get_bundle, get_latest_output, get_model_options, run_forecast
from fnb_forecast.exceptions import ForecastingDomainError

st.set_page_config(
    page_title="F&B Forecasting · Trung tâm quản lý & dự báo",
    page_icon=":material/storefront:",
    layout="wide",
)

inject_custom_css()
output = get_latest_output()

# --- Header Section ---
header_col1, header_col2 = st.columns([3, 1], vertical_alignment="center")
with header_col1:
    st.title(":material/storefront: F&B Forecasting · Trung tâm dự báo kinh doanh")
    st.caption("Hệ thống thông minh F&B: Dự báo nhu cầu món → Doanh thu → Kế hoạch nguyên liệu → Đề xuất đặt hàng")
with header_col2:
    render_provenance(output)

# Display target dates badge
target_dates = output.item_forecasts.get("target_date", pd.Series(dtype="object"))
if not target_dates.empty:
    dates = pd.to_datetime(target_dates).dt.strftime("%d/%m/%Y").drop_duplicates().tolist()
    if dates:
        start_date = dates[0]
        end_date = dates[-1]
        st.info(f"**Giai đoạn dự báo:** Từ ngày **{start_date}** đến **{end_date}** (7 ngày tới)", icon=":material/calendar_today:")

st.write("")

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### :material/tune: Thiết lập dự báo")
    with st.container(border=True):
        dataset_labels = [friendly_dataset_name(k) for k in DATASET_OPTIONS.keys()]
        current_prov = output.metadata.get("provenance", output.metadata.get("dataset_name", DEFAULT_DATASET))
        default_ds_idx = 0
        if len(dataset_labels) > 1 and "real_store" not in str(current_prov):
            default_ds_idx = 1
        
        selected_label = st.selectbox(
            "Bộ dữ liệu kinh doanh",
            dataset_labels,
            index=default_ds_idx,
            help="Chọn dữ liệu thực tế của quán để lập kế hoạch."
        )
        dataset_key = list(DATASET_OPTIONS.keys())[dataset_labels.index(selected_label)]
        
        # Model selection dropdown
        model_options = get_model_options(dataset_key)
        model_labels = list(model_options.values())
        model_keys = list(model_options.keys())
        
        current_req_model = output.metadata.get("requested_model", "auto")
        default_model_idx = model_keys.index(current_req_model) if current_req_model in model_keys else 0

        selected_model_label = st.selectbox(
            "Mô hình dự báo",
            model_labels,
            index=default_model_idx,
            help="Chọn thuật toán toán học hoặc mô hình máy học để thực hiện dự báo."
        )
        selected_model_id = model_keys[model_labels.index(selected_model_label)]

        # Determine calendar date bounds for forecast origin
        bundle = get_bundle(dataset_key)
        min_date_val = date(2025, 5, 1)
        max_date_val = date(2026, 6, 30)
        if bundle is not None and not bundle.calendar.empty and "date" in bundle.calendar.columns:
            try:
                cal_dates = pd.to_datetime(bundle.calendar["date"])
                cal_max_origin = (cal_dates.max() - pd.Timedelta(7, unit="D")).date()
                min_date_val = cal_dates.min().date()
                if not bundle.daily_sales.empty and "date" in bundle.daily_sales.columns:
                    sales_max = pd.to_datetime(bundle.daily_sales["date"]).max().date()
                    max_date_val = min(sales_max, cal_max_origin)
                else:
                    max_date_val = cal_max_origin
            except (TypeError, ValueError):
                pass

        default_origin = output.metadata.get("forecast_origin")
        try:
            if default_origin and isinstance(default_origin, str):
                origin_value = pd.Timestamp(default_origin.split(" ")[0]).date()
            elif default_origin:
                origin_value = pd.Timestamp(default_origin).date()
            else:
                origin_value = max_date_val
        except (TypeError, ValueError):
            origin_value = max_date_val

        if origin_value > max_date_val:
            origin_value = max_date_val
        elif origin_value < min_date_val:
            origin_value = min_date_val

        origin = st.date_input(
            "Ngày chốt dữ liệu (Bắt đầu dự báo)",
            value=origin_value,
            min_value=min_date_val,
            max_value=max_date_val,
            help=f"Mốc thời gian bắt đầu dự báo 7 ngày tiếp theo (cho phép từ {min_date_val.strftime('%d/%m/%Y')} đến {max_date_val.strftime('%d/%m/%Y')})."
        )
        
        service_level_map = {
            0.95: "95%",
            0.98: "98%",
            0.90: "90%",
        }
        sl_label = st.selectbox(
            "Mức độ đảm bảo hàng kho",
            list(service_level_map.values()),
            index=0,
            help="Độ tin cậy để dự trữ hàng an toàn, tránh hết nguyên liệu lúc cao điểm."
        )
        service_level = [k for k, v in service_level_map.items() if v == sl_label][0]
        
        if st.button("Chạy dự báo & lập kế hoạch", type="primary", width="stretch", icon=":material/play_arrow:"):
            try:
                with st.spinner("Đang tính toán dự báo và lập kế hoạch mua hàng..."):
                    output = run_forecast(
                        pd.Timestamp(origin),
                        float(service_level),
                        dataset=dataset_key,
                        model_id=selected_model_id,
                    )
                st.toast("Đã cập nhật xong dữ liệu dự báo cho 7 ngày tới!", icon=":material/check_circle:")
                st.rerun()
            except (ForecastingDomainError, OSError, ValueError, TypeError) as error:
                render_domain_error(error)

# --- KPI Cards Row ---
revenue, cups, needed_ings, alerts = item_metrics(output)

with st.container(horizontal=True):
    st.metric(
        label="Doanh thu dự kiến",
        value=vnd(revenue),
        help="Tổng doanh thu ước tính bán hàng trong 7 ngày tới",
        border=True,
    )
    st.metric(
        label="Số ly dự kiến",
        value=f"{cups:,.0f} ly",
        help="Tổng số lượng ly đồ uống / món ăn dự kiến bán ra",
        border=True,
    )
    st.metric(
        label="Nguyên liệu cần đặt",
        value=f"{needed_ings} loại",
        delta=f"Cần tạo đơn" if needed_ings > 0 else "Đã đủ hàng",
        delta_color="off" if needed_ings > 0 else "normal",
        help="Số lượng nguyên vật liệu cần gửi đơn đặt hàng mới",
        border=True,
    )
    st.metric(
        label="Cảnh báo kho",
        value=f"{alerts} cảnh báo",
        delta="Cần kiểm tra" if alerts > 0 else "An toàn",
        delta_color="inverse" if alerts > 0 else "normal",
        help="Cảnh báo về hạn sử dụng hoặc nguy cơ thiếu hụt",
        border=True,
    )

st.write("")

# --- Content Grid ---
col_left, col_right = st.columns([1, 1])

with col_left:
    with st.container(border=True):
        st.subheader(":material/priority_high: Đề xuất hành động ngay")
        warnings_list = top_warnings(output)
        if warnings_list:
            for warning in warnings_list:
                st.warning(warning, icon=":material/notification_important:")
        else:
            st.success("Tồn kho hiện tại đang rất an toàn, chưa cần tạo thêm đơn đặt hàng khẩn cấp.", icon=":material/check_circle:")
        
        total_orders_needed = int((output.order_proposals["suggested_order"].fillna(0) > 0).sum()) if not output.order_proposals.empty and "suggested_order" in output.order_proposals else 0
        if total_orders_needed > len(warnings_list):
            st.caption(f"Đang hiển thị {len(warnings_list)}/{total_orders_needed} đề xuất ưu tiên hàng đầu. Xem đầy đủ tại **Trang 2 - Kế hoạch nguyên liệu & đặt hàng**.")
        else:
            st.caption("Đề xuất tự động tính toán dựa trên mức tồn kho thực tế và lượng tiêu thụ 7 ngày tới.")

with col_right:
    with st.container(border=True):
        st.subheader(":material/bar_chart: Doanh thu dự kiến 7 ngày tới")
        summary = output.revenue_forecast
        if not summary.empty and {"target_date", "forecast_revenue"}.issubset(summary.columns):
            daily = summary.groupby("target_date", as_index=False)["forecast_revenue"].sum()
            daily["date_dt"] = pd.to_datetime(daily["target_date"])
            daily["day_label"] = daily["date_dt"].dt.strftime("%d/%m (%a)")
            
            try:
                import plotly.express as px
                fig = px.bar(
                    daily,
                    x="day_label",
                    y="forecast_revenue",
                    labels={"day_label": "Ngày", "forecast_revenue": "Doanh thu (₫)"},
                    text_auto=".2s",
                )
                fig.update_traces(
                    marker_color="#0D9488",
                    marker_line_color="#0F766E",
                    marker_line_width=1,
                    opacity=0.9,
                    hovertemplate="Ngày: %{x}<br>Doanh thu: <b>%{y:,.0f} ₫</b><extra></extra>",
                )
                fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, title=None),
                    yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Doanh thu (₫)"),
                )
                st.plotly_chart(fig, width="stretch")
            except ImportError:
                st.bar_chart(daily.set_index("day_label")["forecast_revenue"], width="stretch")
        else:
            table_or_message(summary, "Chưa có dữ liệu doanh thu.")

st.write("")

# --- Top Best Sellers & Quick Breakdown ---
if not output.item_forecasts.empty and "yhat" in output.item_forecasts.columns:
    with st.container(border=True):
        st.subheader(":material/star: Top 5 món bán chạy nhất tuần tới")
        items_summary = output.item_forecasts.copy()
        group_col = "item_name" if "item_name" in items_summary.columns else "item_id"
        top5 = (
            items_summary.groupby(group_col, as_index=False)
            .agg({"yhat": "sum"})
            .sort_values("yhat", ascending=False)
            .head(5)
        )
        
        top_cols = st.columns(min(len(top5), 5))
        for idx, (_, row) in enumerate(top5.iterrows()):
            with top_cols[idx]:
                st.metric(
                    label=f"#{idx+1} {str(row[group_col])}",
                    value=f"{row['yhat']:,.0f} ly",
                    border=True
                )

st.write("")

# --- Export & Audit Tabs ---
with st.container(border=True):
    tab_downloads, tab_audit = st.tabs([
        ":material/download: Xuất báo cáo & dữ liệu",
        ":material/fact_check: Nhật ký chi tiết",
    ])
    
    with tab_downloads:
        st.markdown("**Tải dữ liệu dự báo để quản lý và gửi báo cáo nội bộ:**")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            render_download(
                "Tải file dự báo số lượng món (CSV)", output.item_forecasts, "du_bao_mon.csv", output.warnings
            )
        with d_col2:
            render_download(
                "Tải file đề xuất đặt hàng (CSV)", output.order_proposals, "de_xuat_dat_hang.csv", output.warnings
            )
            
    with tab_audit:
        render_warnings(output.warnings)

