import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.ui_utils import render_reasoning

def render_model_tab(result, show_advanced):
    m = result["model_selection"]
    st.success(f"**Selected Model: {m['selected_model']}**")
    st.markdown("**Rationale:**")
    st.write(m["explanation"])

    if show_advanced:
        with st.expander("🕵️ View Model Selection Reasoning", expanded=False):
            render_reasoning(m.get("reasoning_steps", []))

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
        fig = go.Figure(result["chart_model_comparison"], skip_invalid=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diagnostic Statistics")
    s = result["statistical"]
    diag_df = pd.DataFrame({
        "Metric": [
            "ADF Statistic", "ADF p-value", "ADF Result",
            "KPSS Statistic", "KPSS p-value", "KPSS Result",
            "Trend Detected", "Trend Slope",
            "Seasonal Period", "Dominant Period",
        ],
        "Value": [
            f"{s['adf_statistic']:.4f}",
            f"{s['adf_p_value']:.4f}",
            "Stationary" if s["is_stationary_adf"] else "Non-stationary",
            f"{s['kpss_statistic']:.4f}",
            f"{s['kpss_p_value']:.4f}",
            "Stationary" if s["is_stationary_kpss"] else "Non-stationary",
            "Yes" if s["has_trend"] else "No",
            f"{s['trend_slope']:.6f}",
            str(s["seasonal_period"]) if s.get("seasonal_period") else "None detected",
            f"{s['dominant_period']:.2f}" if s.get("dominant_period") else "N/A",
        ],
        "Interpretation": [
            "More negative = stronger evidence against unit root",
            "< 0.05 → reject unit root (stationary)",
            "ADF conclusion",
            "Lower = more likely stationary",
            "< 0.05 → reject stationarity (non-stationary)",
            "KPSS conclusion",
            "Significant linear trend present",
            "Rate of change per period",
            "Seasonal cycle length (periods per season)",
            "Strongest frequency from periodogram",
        ],
    })
    st.dataframe(diag_df, use_container_width=True, hide_index=True)