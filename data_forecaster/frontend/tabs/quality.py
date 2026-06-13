import streamlit as st
from utils.ui_utils import render_reasoning


def render_quality_tab(result, show_advanced):
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

    if show_advanced:
        with st.expander("🕵️ View Data Validation Reasoning", expanded=False):
            render_reasoning(v.get("reasoning_steps", []))

    st.markdown("**Validation Summary:**")
    st.write(v["summary"])
