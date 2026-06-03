import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go

def render_overview_tab(info, result, uploaded_file):
    st.subheader("Dataset Preview")
    if info:
        try:
            file_bytes = uploaded_file.getvalue() if uploaded_file else None
            if file_bytes:
                if uploaded_file.name.endswith(".csv"):
                    preview_df = pd.read_csv(io.BytesIO(file_bytes)).head(20)
                else:
                    preview_df = pd.read_excel(io.BytesIO(file_bytes)).head(20)
                st.dataframe(preview_df, use_container_width=True)
        except Exception:
            st.info("Preview unavailable.")

    st.subheader("Historical Time Series")
    if result.get("chart_historical"):
        fig = go.Figure(result["chart_historical"], skip_invalid=True)
        st.plotly_chart(fig, use_container_width=True)