from __future__ import annotations

import base64
import json
import os
import re
from typing import Any
import time as _time
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from api_service import ForecastingAPI
from utils.pdf_utils import report_to_pdf
from utils.ui_utils import render_reasoning, preflight_defaults, render_preflight_contents, render_preflight_dialog_content
from tabs.overview import render_overview_tab
from tabs.quality import render_quality_tab
from tabs.stats import render_stats_tab
from tabs.model import render_model_tab
from tabs.forecast import render_forecast_tab
from tabs.report import render_report_tab
from tabs.trace import render_trace_tab
from tabs.chat import render_chat_tab

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Time Series Data Forecaster Agent", layout="wide", page_icon="📈")
st.title("📈 Time Series Data Forecaster Agent")

# ── Session state initialisation ──────────────────────────────────────────────
for key in (
    "upload_info",
    "analysis_result",
    "error",
    "_running",
    "_job_id",
    "_job_progress",
    "_job_step",
    "_user_prompt",
    "_preflight_options",
    "_preflight_options_current",
    "_preflight_signature",
    "_show_preflight_fallback",
    "chat_history",
):
    if key not in st.session_state:
        st.session_state[key] = None


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

if st.session_state.chat_history is None:
    st.session_state.chat_history = []

def _render_preflight_dialog(preflight: dict[str, Any], disabled: bool = False) -> bool:
    return render_preflight_dialog_content(preflight, disabled)


if _dialog:
    @_dialog("Preflight Review")
    def _preflight_dialog(preflight: dict[str, Any], disabled: bool = False) -> None:
        _render_preflight_dialog(preflight, disabled=disabled)
else:
    _preflight_dialog = None


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
                    resp = ForecastingAPI.upload_file(
                        "sample_airline_passengers.csv", _demo_bytes, "text/csv"
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
                    st.error(f"🌐 Backend Connection Error: {exc}. Verify BACKEND_URL and service names in docker-compose.")

    if not demo_clicked:
        st.session_state["_demo_loaded"] = False

    if uploaded_file and (
        st.session_state.upload_info is None
        or st.session_state.upload_info.get("filename") != uploaded_file.name
    ):
        with st.spinner("Uploading…"):
            try:
                resp = ForecastingAPI.upload_file(
                    uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type
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

    model_choice = st.selectbox(
        "Forecasting Model",
        options=["Auto (AI selects)", "Holt-Winters", "ARIMA", "SARIMA"],
        index=0,
        disabled=not info,
        help="Auto lets the AI agent choose the best model. Selecting a model skips that step for faster results.",
    )
    forced_model = None if model_choice == "Auto (AI selects)" else model_choice

    st.markdown("---")
    user_prompt = st.text_area(
        "Business Context / Report Focus (optional)",
        placeholder="e.g. Focus recommendations on inventory planning. Flag any risk of over-forecasting.",
        height=100,
        disabled=not info,
        help="Appended to the AI report prompt so it can tailor the analysis to your needs.",
    )
    show_advanced = True

    is_running = st.session_state._running is True
    preflight = None
    preflight_options = {}
    preflight_blocks_run = False

    if info and date_col and value_col:
        signature = f"{info['file_id']}|{date_col}|{value_col}|{forecast_horizon}"
        if st.session_state._preflight_signature != signature:
            st.session_state._preflight_signature = signature
            st.session_state._preflight_options_current = None

        try:
            resp = ForecastingAPI.get_preflight(
                info["file_id"],
                forecast_horizon,
                date_col,
                value_col
            )
            if resp.status_code == 200:
                preflight = resp.json()
                decisions = preflight.get("decisions", [])
                saved_options = st.session_state.get("_preflight_options_current")
                preflight_options = saved_options or preflight_defaults(preflight)

                if preflight["status"] != "ready":
                    st.markdown("---")
                    st.subheader("Preflight Review")

                    if preflight["status"] == "warning":
                        st.warning("Ready to run with cautions.")
                    elif saved_options:
                        st.success("Preflight choices applied.")
                    else:
                        st.warning(f"{len(decisions)} preflight choice(s) need review.")

                    if preflight.get("warnings") and not decisions:
                        st.caption(f"{len(preflight['warnings'])} caution(s) found.")

                    review_label = "Review Preflight Options" if decisions else "View Preflight Details"
                    if st.button(
                        review_label,
                        disabled=is_running,
                        use_container_width=True,
                    ):
                        if _preflight_dialog:
                            _preflight_dialog(preflight, disabled=is_running)
                        else:
                            st.session_state["_show_preflight_fallback"] = True

                    if not _preflight_dialog and st.session_state.get("_show_preflight_fallback"):
                        with st.expander("Preflight Review", expanded=True):
                            choices = render_preflight_contents(preflight, disabled=is_running)
                            if st.button("Apply Preflight Choices", disabled=is_running, use_container_width=True):
                                st.session_state._preflight_options_current = choices
                                st.session_state["_show_preflight_fallback"] = False
                                st.rerun()

                if preflight_options.get("continue_short_series") == "stop":
                    preflight_blocks_run = True
                    st.info("Run Analysis is paused until short-series confirmation is set to continue.")
            else:
                st.error(resp.json().get("detail", "Preflight review failed."))
                preflight_blocks_run = True
        except Exception as exc:
            st.error(f"Preflight review error: {exc}")
            preflight_blocks_run = True

    run_btn = st.button(
        "⏳ Running…" if is_running else "🚀 Run Analysis",
        disabled=not info or is_running or preflight_blocks_run,
        use_container_width=True,
    )

if run_btn and info:
    st.session_state._running = True
    st.session_state._job_id = None
    st.session_state._job_progress = 0
    st.session_state._job_step = "Submitting job…"
    st.session_state._user_prompt = user_prompt or None
    st.session_state._preflight_options = preflight_options
    st.rerun()

if st.session_state._running and info:
    # ── Progress display ──────────────────────────────────────────────────────
    pct = st.session_state._job_progress or 0
    step_text = st.session_state._job_step or "Processing…"
    st.progress(pct / 100, text=f"{step_text} ({pct}%)")

    job_id = st.session_state._job_id

    if job_id is None:
        # ── Submit the job ────────────────────────────────────────────────────
        try:
            payload = {
                "file_id": info["file_id"],
                "forecast_horizon": forecast_horizon,
                "date_col": date_col,
                "value_col": value_col,
                "forced_model": forced_model,
                "user_prompt": st.session_state.get("_user_prompt"),
                "preflight_options": st.session_state.get("_preflight_options"),
            }
            resp = ForecastingAPI.submit_analysis(payload)
            
            if resp.status_code == 202:
                st.session_state._job_id = resp.json()["job_id"]
                st.session_state._job_progress = 0
                st.session_state._job_step = "Queued — waiting for an available slot…"
            else:
                st.session_state.error = resp.json().get("detail", "Failed to submit job.")
                st.session_state._running = False
        except Exception as exc:
            st.session_state.error = str(exc)
            st.session_state._running = False
        st.rerun()
    else:
        # ── Poll for status ───────────────────────────────────────────────────
        import time as _time
        try:
            resp = ForecastingAPI.get_job_status(job_id)
            if resp.status_code == 200:
                job = resp.json()
                st.session_state._job_progress = job["progress"]
                st.session_state._job_step = job["step"]

                if job["status"] == "done":
                    st.session_state.analysis_result = job["result"]
                    st.session_state.error = None
                    st.session_state._running = False
                    st.session_state._job_id = None
                    st.rerun()
                elif job["status"] == "error":
                    st.session_state.error = job.get("error", "Analysis failed.")
                    st.session_state._running = False
                    st.session_state._job_id = None
                    st.rerun()
                else:
                    # pending or running — poll again after a short delay
                    _time.sleep(1.5)
                    st.rerun()
            else:
                st.session_state.error = "Failed to poll job status."
                st.session_state._running = False
                st.session_state._job_id = None
                st.rerun()
        except Exception as exc:
            st.session_state.error = str(exc)
            st.session_state._running = False
            st.session_state._job_id = None
            st.rerun()

if st.session_state.error and not st.session_state.analysis_result:
    st.error(f"Error: {st.session_state.error}")

# ── Main area — 6 tabs ────────────────────────────────────────────────────────
result = st.session_state.analysis_result

if result:
    tab_report, tab_forecast, tab_model, tab_stats, tab_quality, tab_trace, tab_overview, tab_chat = st.tabs([
        "📄 Report",
        "🔮 Forecast",
        "🤖 Model Selection",
        "📐 Statistical Analysis",
        "🔍 Data Quality",
        "🕵️ AI Reasoning Trace",
        "📊 Data Overview",
        "💬 Chat with your data",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_overview:
        render_overview_tab(info, result, uploaded_file)

    # ── Tab 2: Data Quality ───────────────────────────────────────────────────
    with tab_quality:
        render_quality_tab(result, show_advanced)

    # ── Tab 3: Statistical Analysis ───────────────────────────────────────────
    with tab_stats:
        render_stats_tab(result, show_advanced)

    # ── Tab 4: Model Selection ────────────────────────────────────────────────
    with tab_model:
        render_model_tab(result, show_advanced)

    # ── Tab 5: Forecast ───────────────────────────────────────────────────────
    with tab_forecast:
        render_forecast_tab(result)

    # ── Tab 6: Report ─────────────────────────────────────────────────────────
    with tab_report:
        render_report_tab(result, info)

    # ── Tab 7: AI Trace ───────────────────────────────────────────────────────
    with tab_trace:
        render_trace_tab(result)

    # ── Tab 8: Data Explorer ──────────────────────────────────────────────────
    with tab_chat:
        render_chat_tab(info)

else:
    st.info("Upload a time series file and click **Run Analysis** to get started.")
