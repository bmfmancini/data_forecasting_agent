import streamlit as st
from utils.pdf_utils import report_to_pdf

def render_report_tab(result, info):
    report_text = result.get("report", "Report not available.")
    try:
        pdf_bytes = report_to_pdf(report_text)
        fname = f"forecast_report_{info['filename'].rsplit('.', 1)[0]}.pdf" if info else "forecast_report.pdf"
        st.download_button(
            label="⬇️ Download Report as PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as _pdf_err:
        st.warning(f"PDF export unavailable: {_pdf_err}")
    st.markdown("---")
    st.markdown(report_text)