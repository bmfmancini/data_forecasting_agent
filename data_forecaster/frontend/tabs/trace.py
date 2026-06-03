import streamlit as st
from utils.ui_utils import render_reasoning

def render_trace_tab(result):
    st.subheader("🕵️ Full AI Reasoning Trace")
    st.write("This log shows the internal 'monologue' and tool usage of the AI agents as they processed your request.")
    
    with st.expander("1. Data Validation Agent", expanded=True):
        render_reasoning(result["validation"].get("reasoning_steps", []))
        
    with st.expander("2. Statistical Analysis Agent", expanded=True):
        render_reasoning(result["statistical"].get("reasoning_steps", []))
        
    with st.expander("3. Model Selection Agent", expanded=True):
        render_reasoning(result["model_selection"].get("reasoning_steps", []))
        
    with st.expander("4. Forecasting Agent", expanded=True):
        render_reasoning(result["forecast"].get("reasoning_steps", []))
        
    with st.expander("5. Report Generation Agent", expanded=True):
        render_reasoning(result.get("report_reasoning", []))
        
    st.info("💡 Advanced Mode (toggle in sidebar) also shows these traces inline within each specific analysis tab.")