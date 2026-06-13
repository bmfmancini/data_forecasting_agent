import streamlit as st
import base64
import plotly.graph_objects as go
from utils.ui_utils import render_reasoning


def render_stats_tab(result, show_advanced):
    s = result["statistical"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Stationarity Tests")
        adf_status = "✅ Stationary" if s["is_stationary_adf"] else "⚠️ Non-stationary"
        kpss_status = (
            "✅ Stationary" if s["is_stationary_kpss"] else "⚠️ Non-stationary"
        )
        st.markdown(
            f"**ADF Test:** {adf_status}  \np-value = `{s['adf_p_value']:.4f}`, statistic = `{s['adf_statistic']:.4f}`"
        )
        st.markdown(
            f"**KPSS Test:** {kpss_status}  \np-value = `{s['kpss_p_value']:.4f}`, statistic = `{s['kpss_statistic']:.4f}`"
        )

    with col2:
        st.markdown("### Trend & Seasonality")
        trend_status = "✅ Detected" if s["has_trend"] else "➖ Not detected"
        st.markdown(f"**Trend:** {trend_status} (slope=`{s['trend_slope']:.6f}`)")
        st.markdown(f"**Seasonal period:** `{s['seasonal_period']}`")
        st.markdown(f"**Dominant periodogram period:** `{s['dominant_period']:.2f}`")

    if show_advanced:
        with st.expander("🕵️ View Statistical Analysis Reasoning", expanded=False):
            render_reasoning(s.get("reasoning_steps", []))

    st.markdown("**Statistical Summary:**")
    st.write(s["summary"])

    st.subheader("STL Decomposition")
    if result.get("chart_stl"):
        fig = go.Figure(result["chart_stl"], skip_invalid=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("STL chart unavailable.")

    st.subheader("ACF / PACF")
    if result.get("chart_acf_pacf"):
        img_bytes = base64.b64decode(result["chart_acf_pacf"])
        st.image(img_bytes, use_column_width=True)
    else:
        st.info("ACF/PACF chart unavailable.")
