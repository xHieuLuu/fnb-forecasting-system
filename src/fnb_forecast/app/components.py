"""Reusable visual and export helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from fnb_forecast.exceptions import DataValidationError, ForecastingDomainError
from fnb_forecast.planning.revenue import PlanningOutput

__all__ = [
    "inject_custom_css",
    "friendly_dataset_name",
    "friendly_model_name",
    "render_provenance",
    "render_warnings",
    "vnd",
    "render_interval_chart",
    "render_download",
    "render_metadata_download",
    "render_domain_error",
    "table_or_message",
    "top_warnings",
    "item_metrics",
    "interval_totals",
    "render_model_comparison_chart",
]


def inject_custom_css() -> None:
    """Inject modern capstone UI styling for container borders, badges, metrics and sidebar."""
    st.markdown(
        """
        <style>
        /* Modern Container Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px !important;
            border: 1px solid #E2E8F0 !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
            padding: 14px !important;
        }
        
        /* Metric Cards Polish */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #CBD5E1 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            color: #64748B !important;
        }
        div[data-testid="stMetricValue"] {
            font-weight: 700 !important;
            color: #0F172A !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            border-right: 1px solid #E2E8F0 !important;
            background-color: #F1F5F9 !important;
        }

        /* Tab Navigation Polish */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding-top: 8px !important;
            padding-bottom: 8px !important;
        }

        /* Status Badges */
        .capstone-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .chip-teal {
            background-color: #CCFBF1;
            color: #0F766E;
            border: 1px solid #99F6E4;
        }
        .chip-amber {
            background-color: #FEF3C7;
            color: #B45309;
            border: 1px solid #FDE68A;
        }
        .chip-slate {
            background-color: #F1F5F9;
            color: #475569;
            border: 1px solid #E2E8F0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def friendly_dataset_name(dataset: str | None) -> str:
    if not dataset:
        return "Quán thực tế (14 tháng)"
    if "real_store" in dataset:
        return "Quán thực tế (14 tháng - 2025/2026)"
    if "m5" in dataset:
        return "Benchmark M5"
    return str(dataset)


def friendly_model_name(model_id: str | None) -> str:
    if not model_id:
        return "Mô hình dự báo chuẩn"
    name = str(model_id)
    if "lgb" in name.lower() or "lightgbm" in name.lower():
        return f"Mô hình cây quyết định LightGBM ({name})"
    if "tft" in name.lower():
        return f"Mô hình học sâu TFT ({name})"
    if "lstm" in name.lower():
        return f"Mô hình mạng nơ-ron LSTM ({name})"
    if "ets" in name.lower():
        return f"Mô hình san bằng hàm mũ ETS ({name})"
    if "seasonal_naive" in name.lower():
        return "Mô hình chu kỳ tuần (Seasonal Naive)"
    return name


def render_provenance(output: PlanningOutput) -> None:
    metadata = output.metadata
    provenance = friendly_dataset_name(str(metadata.get("provenance", metadata.get("dataset_name", "real_store_14m"))))
    origin = metadata.get("forecast_origin", "Chưa có ngày")
    if isinstance(origin, str) and " " in origin:
        origin = origin.split(" ")[0]
    horizon = metadata.get("horizon_days", 7)
    actual_model = friendly_model_name(metadata.get("actual_model", metadata.get("requested_model", "Mô hình tự động")))
    raw_prov = str(metadata.get("provenance", metadata.get("dataset_name", "real_store_14m")))

    with st.popover("Thông tin đợt dự báo", icon=":material/info:"):
        st.info(f"Nguồn dữ liệu: `{provenance}` (`{raw_prov}`) · Ngày chốt: `{origin}` · Chu kỳ: `{horizon} ngày`", icon=":material/info:")
        st.markdown(f"**Thuật toán vận hành:** {actual_model}")
        if metadata.get("fallback_used"):
            st.info(
                f"Đang kích hoạt chế độ dự phòng an toàn: `{actual_model}`.",
                icon=":material/shield:"
            )


def render_warnings(warnings: list[str]) -> None:
    if not warnings:
        st.success("Tất cả chỉ số đều an toàn, không có cảnh báo nào.", icon=":material/check_circle:")
        return
    for warning in warnings:
        msg = str(warning)
        if "weather unavailable" in msg.lower():
            st.info("Đang dùng dữ liệu lịch sử chu kỳ để dự báo do chưa kết nối trạm thời tiết trực tiếp.", icon=":material/info:")
        elif "safety stock unavailable" in msg.lower():
            st.info("Tồn kho an toàn đang áp dụng mức tối thiểu mặc định.", icon=":material/info:")
        else:
            st.warning(msg, icon=":material/warning:")


def vnd(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0 ₫"
    return f"{float(value):,.0f} ₫"


def render_interval_chart(frame: pd.DataFrame, *, value_column: str = "yhat") -> None:
    required = {"target_date", value_column}
    if frame.empty or not required.issubset(frame.columns):
        st.info("Chưa có đủ dữ liệu để vẽ biểu đồ.", icon=":material/info:")
        return
    chart = frame.copy()
    chart["target_date"] = pd.to_datetime(chart["target_date"], errors="coerce")
    chart = chart.dropna(subset=["target_date"]).sort_values("target_date")

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.line_chart(chart.set_index("target_date")[[c for c in ["actual", value_column] if c in chart]])
        return

    fig = go.Figure()
    dates = chart["target_date"].dt.strftime("%d/%m (%a)")

    # Upper & Lower bounds (Confidence Interval Area)
    if "upper" in chart.columns and "lower" in chart.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=chart["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Tối đa",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=chart["lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(13, 148, 136, 0.15)",
                name="Khoảng an toàn (Tối thiểu – Tối đa)",
                hoverinfo="skip",
            )
        )

    # Actual series if present
    if "actual" in chart.columns and chart["actual"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=chart["actual"],
                mode="lines+markers",
                name="Thực tế đã bán",
                line=dict(color="#64748B", width=2, dash="dot"),
                marker=dict(size=6, color="#64748B"),
                hovertemplate="Thực tế: %{y:,.1f} ly<extra></extra>",
            )
        )

    # Forecast series
    if value_column in chart.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=chart[value_column],
                mode="lines+markers",
                name="Dự báo bán ra",
                line=dict(color="#0D9488", width=3),
                marker=dict(size=7, color="#0F766E"),
                hovertemplate="Dự báo: <b>%{y:,.1f} ly</b><extra></extra>",
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            title=None,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            title="Số lượng (ly / món)",
        ),
    )
    st.plotly_chart(fig, width="stretch")


def render_download(
    label: str,
    frame: pd.DataFrame,
    file_name: str,
    warnings: list[str] | None = None,
) -> None:
    downloadable = frame.copy()
    if warnings:
        downloadable["luu_y_he_thong"] = " | ".join(str(value) for value in warnings)
    st.download_button(
        label,
        data=downloadable.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
    )


def render_metadata_download(output: PlanningOutput) -> None:
    payload = json.dumps(output.metadata, ensure_ascii=False, indent=2, default=str)
    st.download_button(
        "Tải thông số kỹ thuật (JSON)",
        data=payload.encode("utf-8"),
        file_name="thong_so_du_bao.json",
        mime="application/json",
        icon=":material/code:",
    )


def render_domain_error(error: Exception) -> None:
    if isinstance(error, DataValidationError):
        st.error(
            f"Dữ liệu chưa đúng quy chuẩn — bảng `{error.table}`, cột `{', '.join(error.columns)}`: {error.reason}",
            icon=":material/error:",
        )
    elif isinstance(error, ForecastingDomainError):
        st.error(str(error), icon=":material/error:")
    else:
        st.exception(error)


def table_or_message(frame: pd.DataFrame, message: str = "Chưa có dữ liệu.") -> None:
    if frame.empty:
        st.info(message, icon=":material/info:")
        return

    # Create intuitive Vietnamese column configs
    column_config: dict[str, Any] = {}
    display_frame = frame.copy()

    # Define column configurations
    for col in display_frame.columns:
        if col in ("forecast_revenue", "revenue", "cost", "unit_price", "total_cost", "total_price"):
            column_config[col] = st.column_config.NumberColumn("Doanh thu / Giá (₫)" if "price" in col or "rev" in col else "Chi phí (₫)", format="%.0f ₫")
        elif col in ("yhat", "forecast_qty", "quantity", "sales"):
            column_config[col] = st.column_config.NumberColumn("Số lượng dự báo (ly)", format="%.1f")
        elif col in ("lower",):
            column_config[col] = st.column_config.NumberColumn("Tối thiểu (ly)", format="%.1f")
        elif col in ("upper",):
            column_config[col] = st.column_config.NumberColumn("Tối đa (ly)", format="%.1f")
        elif col in ("forecast_need",):
            column_config[col] = st.column_config.NumberColumn("Nhu cầu cần dùng", format="%.2f")
        elif col in ("coverage_forecast_need",):
            column_config[col] = st.column_config.NumberColumn("Nhu cầu tiêu thụ (chu kỳ)", format="%.2f")
        elif col in ("usable_on_hand",):
            column_config[col] = st.column_config.NumberColumn("Tồn kho khả dụng (FEFO)", format="%.2f")
        elif col in ("on_hand",):
            column_config[col] = st.column_config.NumberColumn("Tồn kho thực tế", format="%.2f")
        elif col in ("safety_stock",):
            column_config[col] = st.column_config.NumberColumn("Tồn kho an toàn", format="%.2f")
        elif col in ("scheduled_receipts",):
            column_config[col] = st.column_config.NumberColumn("Hàng đang về", format="%.2f")
        elif col in ("target_stock",):
            column_config[col] = st.column_config.NumberColumn("Tồn kho mục tiêu", format="%.2f")
        elif col in ("raw_order",):
            column_config[col] = st.column_config.NumberColumn("Lượng thiếu hụt thô", format="%.2f")
        elif col in ("suggested_order", "order_qty"):
            column_config[col] = st.column_config.NumberColumn("Đề xuất đặt hàng", format="%.2f")
        elif col == "shelf_life_cap_applied":
            column_config[col] = st.column_config.CheckboxColumn("Áp dụng hạn dùng")
        elif col == "waste_risk_warning":
            column_config[col] = st.column_config.CheckboxColumn("Rủi ro cận date")
        elif col == "lot_id":
            column_config[col] = st.column_config.TextColumn("Mã lô hàng")
        elif col in ("received_date", "receipt_date"):
            column_config[col] = st.column_config.DateColumn("Ngày nhập kho", format="YYYY-MM-DD")
        elif col in ("expiry_date", "expiration_date"):
            column_config[col] = st.column_config.DateColumn("Hạn sử dụng", format="YYYY-MM-DD")
        elif col in ("coverage_days",):
            column_config[col] = st.column_config.NumberColumn("Số ngày chu kỳ", format="%d")
        elif col in ("rmse",):
            column_config[col] = st.column_config.NumberColumn("Sai số RMSE", format="%.2f")
        elif "date" in col:
            column_config[col] = st.column_config.DateColumn("Ngày", format="YYYY-MM-DD")
        elif col == "item_name":
            column_config[col] = st.column_config.TextColumn("Tên món")
        elif col == "item_id":
            column_config[col] = st.column_config.TextColumn("Mã món")
        elif col == "category":
            column_config[col] = st.column_config.TextColumn("Danh mục")
        elif col in ("ingredient_name", "name"):
            column_config[col] = st.column_config.TextColumn("Tên nguyên liệu")
        elif col == "ingredient_id":
            column_config[col] = st.column_config.TextColumn("Mã nguyên liệu")
        elif col in ("unit", "base_unit"):
            column_config[col] = st.column_config.TextColumn("Đơn vị")
        elif col == "pack_size":
            column_config[col] = st.column_config.NumberColumn("Quy cách gói", format="%.0f")
        elif col in ("warning", "status"):
            column_config[col] = st.column_config.TextColumn("Trạng thái / Ghi chú")
        elif col == "model_id":
            column_config[col] = st.column_config.TextColumn("Mã mô hình")
        elif "wape" in col:
            if pd.api.types.is_numeric_dtype(display_frame[col]) and not display_frame[col].dropna().empty:
                if display_frame[col].dropna().abs().max() <= 10.0:
                    display_frame[col] = display_frame[col] * 100.0
            column_config[col] = st.column_config.NumberColumn("Sai số WAPE", format="%.2f%%")
        elif "mae" in col:
            column_config[col] = st.column_config.NumberColumn("Sai số MAE (ly)", format="%.2f")
        elif "bias" in col:
            column_config[col] = st.column_config.NumberColumn("Độ lệch Bias", format="%.3f")

    if "model_id" in display_frame.columns and "model_name" not in display_frame.columns and "item_name" not in display_frame.columns:
        display_frame["Ten_Mo_Hinh"] = display_frame["model_id"].apply(friendly_model_name)
        column_config["Ten_Mo_Hinh"] = st.column_config.TextColumn("Thuật toán / Mô hình")

    # If general business table with item/ingredient data, hide debug columns
    debug_cols = {
        "actual_model", "requested_model", "fallback_used", "fallback_reason",
        "forecast_origin", "dataset_checksum", "service_level"
    }
    if "item_name" in display_frame.columns or "ingredient_name" in display_frame.columns:
        display_frame = display_frame[[c for c in display_frame.columns if c not in debug_cols and c != "model_id"]]

    # Order columns nicely if friendly columns are present
    preferred_order = [
        "Ten_Mo_Hinh", "model_id", "target_date", "item_name", "item_id", "category", "yhat", "lower", "upper", "unit_price", "forecast_revenue",
        "ingredient_name", "ingredient_id", "unit", "base_unit", "forecast_need", "coverage_forecast_need",
        "usable_on_hand", "safety_stock", "scheduled_receipts", "target_stock", "raw_order", "suggested_order",
        "pack_size", "shelf_life_cap_applied", "waste_risk_warning", "warning", "lot_id", "on_hand", "received_date", "expiry_date"
    ]
    present_cols = [c for c in preferred_order if c in display_frame.columns] + [c for c in display_frame.columns if c not in preferred_order]
    
    st.dataframe(
        display_frame[present_cols],
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


def top_warnings(output: PlanningOutput, limit: int = 4) -> list[str]:
    warnings = []
    orders = output.order_proposals
    if not orders.empty:
        for _, row in orders.iterrows():
            name = str(row.get("ingredient_name", row.get("ingredient_id", "")))
            suggested = row.get("suggested_order", 0)
            warning_msg = row.get("warning", "")
            unit = row.get("base_unit", row.get("unit", ""))
            
            if pd.notna(suggested) and float(suggested) > 0:
                warnings.append(f"**{name}**: Cần đặt thêm **{float(suggested):,.1f} {unit}** để đảm bảo phục vụ 7 ngày tới.")
            elif warning_msg and pd.notna(warning_msg) and str(warning_msg).strip():
                warnings.append(f"**{name}**: {warning_msg}")

    if not warnings and output.warnings:
        for w in output.warnings:
            if "weather unavailable" in str(w).lower():
                continue
            if "safety stock unavailable" in str(w).lower():
                continue
            warnings.append(str(w))

    return list(dict.fromkeys(warnings))[:limit]


def item_metrics(output: PlanningOutput) -> tuple[float, float, int, int]:
    revenue = output.revenue_forecast
    items = output.item_forecasts
    total_revenue = (
        float(revenue["forecast_revenue"].sum())
        if "forecast_revenue" in revenue and not revenue.empty
        else 0.0
    )
    total_cups = float(items["yhat"].sum()) if "yhat" in items and not items.empty else 0.0
    
    needed_ingredients = 0
    orders = output.order_proposals
    alerts = 0
    if not orders.empty:
        if "suggested_order" in orders:
            needed_ingredients = int((orders["suggested_order"].fillna(0) > 0).sum())
        if "warning" in orders:
            alerts += int(orders["warning"].fillna("").astype(str).ne("").sum())
        if "waste_risk_warning" in orders:
            alerts += int(orders["waste_risk_warning"].fillna(False).astype(bool).sum())

    return total_revenue, total_cups, needed_ingredients, alerts


def interval_totals(output: PlanningOutput) -> tuple[float, float, float]:
    frame = output.item_forecasts
    total = float(frame["yhat"].sum()) if "yhat" in frame and not frame.empty else 0.0
    lower = float(frame["lower"].sum()) if "lower" in frame and not frame.empty else total
    upper = float(frame["upper"].sum()) if "upper" in frame and not frame.empty else total
    return total, lower, upper


def render_model_comparison_chart(frame: pd.DataFrame) -> None:
    """Render Plotly bar chart comparing WAPE (%) and MAE across models for evaluation page."""
    if frame.empty or "model_id" not in frame.columns:
        st.info("Chưa có đủ dữ liệu để vẽ biểu đồ so sánh mô hình.", icon=":material/info:")
        return

    chart_df = frame.copy()
    chart_df["friendly_name"] = chart_df["model_id"].apply(friendly_model_name)
    
    # Format WAPE %
    if "validation_wape" in chart_df.columns:
        chart_df["wape_pct"] = chart_df["validation_wape"].apply(
            lambda v: float(v) * 100.0 if float(v) <= 10.0 else float(v)
        )
    elif "wape" in chart_df.columns:
        chart_df["wape_pct"] = chart_df["wape"].apply(
            lambda v: float(v) * 100.0 if float(v) <= 10.0 else float(v)
        )
    else:
        chart_df["wape_pct"] = 0.0

    # Sort by WAPE ascending (best first)
    chart_df = chart_df.sort_values("wape_pct", ascending=True)

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.bar_chart(chart_df.set_index("friendly_name")["wape_pct"])
        return

    fig = go.Figure()
    
    # Add WAPE bar
    fig.add_trace(
        go.Bar(
            x=chart_df["friendly_name"],
            y=chart_df["wape_pct"],
            name="Sai số WAPE (%)",
            marker_color="#0D9488",
            text=[f"{v:.1f}%" for v in chart_df["wape_pct"]],
            textposition="auto",
            hovertemplate="Mô hình: <b>%{x}</b><br>WAPE: <b>%{y:.2f}%%</b><extra></extra>",
        )
    )

    # Add MAE bar if available
    mae_col = "validation_mae" if "validation_mae" in chart_df.columns else ("mae" if "mae" in chart_df.columns else None)
    if mae_col:
        fig.add_trace(
            go.Bar(
                x=chart_df["friendly_name"],
                y=chart_df[mae_col],
                name="Sai số MAE (ly / ngày)",
                marker_color="#6366F1",
                text=[f"{v:.2f}" for v in chart_df[mae_col]],
                textposition="auto",
                hovertemplate="Mô hình: <b>%{x}</b><br>MAE: <b>%{y:.2f} ly</b><extra></extra>",
            )
        )

    fig.update_layout(
        height=400,
        barmode="group",
        margin=dict(l=20, r=20, t=30, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            showgrid=False,
            title=None,
            tickangle=-15,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            title="Chỉ số sai số (WAPE % / MAE ly)",
        ),
    )
    st.plotly_chart(fig, width="stretch")


