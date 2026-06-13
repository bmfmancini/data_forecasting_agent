"""
Report tab module for the Time Series Data Forecaster Agent.
Handles the display of forecast reports and PDF export functionality.
"""

import streamlit as st
from utils.pdf_utils import report_to_pdf


def render_report_tab(result: dict, info: dict) -> None:
    """
    Render the report tab with forecast results and PDF export.

    Args:
        result: Dictionary containing analysis results
        info: Dictionary containing file information
    """
    report_text = result.get("report", "Report not available.")

    # PDF export functionality
    try:
        pdf_bytes = report_to_pdf(report_text)
        fname = (
            f"forecast_report_{info['filename'].rsplit('.', 1)[0]}.pdf"
            if info
            else "forecast_report.pdf"
        )
        st.download_button(
            label="⬇️ Download Report as PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as pdf_error:
        st.warning(f"PDF export unavailable: {pdf_error}")

    st.markdown("---")
    st.markdown(report_text)
