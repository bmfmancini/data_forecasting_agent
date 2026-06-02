from __future__ import annotations

import base64
import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Data Forecaster", layout="wide", page_icon="📈")
st.title("📈 Data Forecaster")

# ── Session state initialisation ──────────────────────────────────────────────
for key in ("upload_info", "analysis_result", "error"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    demo_clicked = st.button("📂 Load Demo Data", use_container_width=True, help="Loads the classic airline passengers dataset (1949–1960)")

    st.markdown("**— or upload your own —**")

    uploaded_file = st.file_uploader(
        "Upload Time Series (CSV or XLSX)", type=["csv", "xlsx"]
    )

    if demo_clicked and not st.session_state.get("_demo_loaded"):
        import io as _io
        try:
            _demo_path = os.path.join(os.path.dirname(__file__), "demo_data.csv")
            if not os.path.exists(_demo_path):
                # Fall back to fetching from backend container's data dir via env path
                _demo_path = "/app/data/sample_airline_passengers.csv"
            with open(_demo_path, "rb") as _f:
                _demo_bytes = _f.read()
        except FileNotFoundError:
            st.error("Demo data file not found inside the container.")
            _demo_bytes = None
        if _demo_bytes:
            with st.spinner("Loading demo data…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": ("sample_airline_passengers.csv", _demo_bytes, "text/csv")},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.session_state.upload_info = resp.json()
                        st.session_state.analysis_result = None
                        st.session_state.error = None
                        st.session_state["_demo_loaded"] = True
                        st.success("Demo data loaded — 144 rows (airline passengers 1949–1960).")
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Demo upload failed."))
                except Exception as exc:
                    st.error(f"Demo load error: {exc}")

    if not demo_clicked:
        st.session_state["_demo_loaded"] = False

    if uploaded_file and (
        st.session_state.upload_info is None
        or st.session_state.upload_info.get("filename") != uploaded_file.name
    ):
        with st.spinner("Uploading…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=60,
                )
                if resp.status_code == 200:
                    st.session_state.upload_info = resp.json()
                    st.session_state.analysis_result = None
                    st.session_state.error = None
                    st.success(f"Uploaded — {st.session_state.upload_info['rows']} rows detected.")
                else:
                    st.session_state.error = resp.json().get("detail", "Upload failed.")
                    st.error(st.session_state.error)
            except Exception as exc:
                st.session_state.error = str(exc)
                st.error(f"Upload error: {exc}")

    info = st.session_state.upload_info
    columns = info["columns"] if info else []

    date_col = st.selectbox(
        "Date Column",
        options=columns,
        index=columns.index(info["detected_date_col"]) if info and info.get("detected_date_col") in columns else 0,
        disabled=not columns,
    )
    value_col = st.selectbox(
        "Value Column",
        options=columns,
        index=columns.index(info["detected_value_col"]) if info and info.get("detected_value_col") in columns else min(1, len(columns) - 1),
        disabled=not columns,
    )

    st.markdown("---")

    freq_label = info["detected_frequency"] if info else "—"
    st.caption(f"Detected frequency: **{freq_label}**")

    forecast_horizon = st.slider(
        "Forecast Horizon (periods)",
        min_value=7, max_value=365, value=12, step=1,
        disabled=not info,
    )

    run_btn = st.button("🚀 Run Analysis", disabled=not info, use_container_width=True)

if run_btn and info:
    with st.spinner("Running 5-agent pipeline — this may take a minute…"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/analyze",
                json={
                    "file_id": info["file_id"],
                    "forecast_horizon": forecast_horizon,
                    "date_col": date_col,
                    "value_col": value_col,
                },
                timeout=600,
            )
            if resp.status_code == 200:
                st.session_state.analysis_result = resp.json()
                st.session_state.error = None
            else:
                st.session_state.error = resp.json().get("detail", "Analysis failed.")
        except Exception as exc:
            st.session_state.error = str(exc)

if st.session_state.error and not st.session_state.analysis_result:
    st.error(f"Error: {st.session_state.error}")

# ── Main area — 6 tabs ────────────────────────────────────────────────────────
result = st.session_state.analysis_result

if result:
    tab_overview, tab_quality, tab_stats, tab_model, tab_forecast, tab_report = st.tabs([
        "📊 Overview",
        "🔍 Data Quality",
        "📐 Statistical Analysis",
        "🤖 Model Selection",
        "🔮 Forecast",
        "📄 Report",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Dataset Preview")
        if info:
            try:
                file_bytes = uploaded_file.getvalue() if uploaded_file else None
                if file_bytes:
                    import io
                    if uploaded_file.name.endswith(".csv"):
                        preview_df = pd.read_csv(io.BytesIO(file_bytes)).head(20)
                    else:
                        preview_df = pd.read_excel(io.BytesIO(file_bytes)).head(20)
                    st.dataframe(preview_df, use_container_width=True)
            except Exception:
                st.info("Preview unavailable.")

        st.subheader("Historical Time Series")
        if result.get("chart_historical"):
            fig = go.Figure(result["chart_historical"])
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Data Quality ───────────────────────────────────────────────────
    with tab_quality:
        v = result["validation"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", v["row_count"])
        col2.metric("Missing Timestamps", v["missing_timestamps"])
        col3.metric("Duplicate Timestamps", v["duplicate_timestamps"])
        col4.metric("Missing Values", v["missing_values"])

        st.markdown(f"**Frequency detected:** `{v['frequency']}`")
        st.markdown(f"**Regular intervals:** {'✅ Yes' if v['is_regular'] else '⚠️ No'}")

        if v["issues"]:
            st.warning("**Issues found:**\n" + "\n".join(f"- {i}" for i in v["issues"]))
        else:
            st.success("No data quality issues detected.")

        st.markdown("**Validation Summary:**")
        st.write(v["summary"])

    # ── Tab 3: Statistical Analysis ───────────────────────────────────────────
    with tab_stats:
        s = result["statistical"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Stationarity Tests")
            adf_status = "✅ Stationary" if s["is_stationary_adf"] else "⚠️ Non-stationary"
            kpss_status = "✅ Stationary" if s["is_stationary_kpss"] else "⚠️ Non-stationary"
            st.markdown(f"**ADF Test:** {adf_status}  \np-value = `{s['adf_p_value']:.4f}`, statistic = `{s['adf_statistic']:.4f}`")
            st.markdown(f"**KPSS Test:** {kpss_status}  \np-value = `{s['kpss_p_value']:.4f}`, statistic = `{s['kpss_statistic']:.4f}`")

        with col2:
            st.markdown("### Trend & Seasonality")
            trend_status = "✅ Detected" if s["has_trend"] else "➖ Not detected"
            st.markdown(f"**Trend:** {trend_status} (slope=`{s['trend_slope']:.6f}`)")
            st.markdown(f"**Seasonal period:** `{s['seasonal_period']}`")
            st.markdown(f"**Dominant periodogram period:** `{s['dominant_period']:.2f}`")

        st.markdown("**Statistical Summary:**")
        st.write(s["summary"])

        st.subheader("STL Decomposition")
        if result.get("chart_stl"):
            fig = go.Figure(result["chart_stl"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("STL chart unavailable.")

        st.subheader("ACF / PACF")
        if result.get("chart_acf_pacf"):
            img_bytes = base64.b64decode(result["chart_acf_pacf"])
            st.image(img_bytes, use_column_width=True)
        else:
            st.info("ACF/PACF chart unavailable.")

    # ── Tab 4: Model Selection ────────────────────────────────────────────────
    with tab_model:
        m = result["model_selection"]
        st.success(f"**Selected Model: {m['selected_model']}**")
        st.markdown("**Rationale:**")
        st.write(m["explanation"])

        rejected = {
            k: v for k, v in {
                "Holt-Winters": m.get("holt_winters_rejected_reason"),
                "ARIMA": m.get("arima_rejected_reason"),
                "SARIMA": m.get("sarima_rejected_reason"),
            }.items() if v
        }
        if rejected:
            st.markdown("**Models not selected:**")
            for model_name, reason in rejected.items():
                st.markdown(f"- **{model_name}:** {reason}")

        st.subheader("Model Comparison Metrics")
        if result.get("chart_model_comparison"):
            fig = go.Figure(result["chart_model_comparison"])
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Forecast ───────────────────────────────────────────────────────
    with tab_forecast:
        f = result["forecast"]

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{f['rmse']:.4f}")
        col2.metric("MAE", f"{f['mae']:.4f}")
        col3.metric("MAPE", f"{f['mape']:.2f}%")

        st.subheader("Forecast Chart")
        if result.get("chart_forecast"):
            fig = go.Figure(result["chart_forecast"])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Values")
        fc_df = pd.DataFrame({
            "Date": f["forecast_dates"],
            "Forecast": [round(v, 4) for v in f["forecast"]],
            "Lower CI (95%)": [round(v, 4) for v in f["lower_ci"]],
            "Upper CI (95%)": [round(v, 4) for v in f["upper_ci"]],
        })
        st.dataframe(fc_df, use_container_width=True)

    # ── Tab 6: Report ─────────────────────────────────────────────────────────
    with tab_report:
        st.markdown(result.get("report", "Report not available."))

else:
    st.info("Upload a time series file and click **Run Analysis** to get started.")
