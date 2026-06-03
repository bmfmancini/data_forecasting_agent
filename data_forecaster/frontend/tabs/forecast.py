import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_forecast_tab(result):
    f = result["forecast"]

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{f['rmse']:.4f}")
    col2.metric("MAE", f"{f['mae']:.4f}")
    col3.metric("MAPE", f"{f['mape']:.2f}%")

    st.subheader("Forecast Chart")
    if result.get("chart_forecast"):
        fig = go.Figure(result["chart_forecast"], skip_invalid=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")
    fc_df = pd.DataFrame({
        "Date": f["forecast_dates"],
        "Forecast": [round(v, 4) for v in f["forecast"]],
        "Lower CI (95%)": [round(v, 4) for v in f["lower_ci"]],
        "Upper CI (95%)": [round(v, 4) for v in f["upper_ci"]],
    })
    st.dataframe(fc_df, use_container_width=True)