from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _report_to_pdf(report_md: str, title: str = "Forecast Report") -> bytes:
    """Render a markdown report string to PDF bytes using fpdf2."""
    from fpdf import FPDF

    def _sanitize(text: str) -> str:
        """Drop chars outside Latin-1 so core fonts don't crash."""
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _strip_inline(text: str) -> str:
        """Remove bold/italic/code markers, keep text content."""
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`(.*?)`", r"\1", text)
        return text

    def _cell(pdf: FPDF, h: int, text: str) -> None:
        """Reset x to left margin then render a multi_cell."""
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, text)

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 12, _sanitize(title), align="C")
    pdf.ln(6)

    for raw_line in report_md.splitlines():
        line = raw_line.rstrip()
        if line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            _cell(pdf, 7, _sanitize(_strip_inline(line[4:])))
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 15)
            _cell(pdf, 8, _sanitize(_strip_inline(line[3:])))
            pdf.ln(2)
        elif line.startswith("# "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 17)
            _cell(pdf, 9, _sanitize(_strip_inline(line[2:])))
            pdf.ln(2)
        elif re.match(r"^[-*] ", line):
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, _sanitize("  - " + _strip_inline(line[2:])))
        elif re.match(r"^\d+\. ", line):
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, _sanitize("  " + _strip_inline(line)))
        elif re.match(r"^-{3,}$", line) or re.match(r"^\*{3,}$", line):
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(4)
        elif line == "":
            pdf.ln(3)
        else:
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, _sanitize(_strip_inline(line)))

    try:
        pdf_output_result = pdf.output(dest='S')
    except TypeError:
        pdf_output_result = pdf.output()

    if isinstance(pdf_output_result, str):
        # If fpdf.output() unexpectedly returns a string, encode it to bytes
        return pdf_output_result.encode("latin-1", errors="replace")
    return pdf_output_result

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Time Data Forecaster Agent", layout="wide", page_icon="📈")
st.title("📈 Time Data Forecaster Agent")

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
):
    if key not in st.session_state:
        st.session_state[key] = None


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def _preflight_defaults(preflight: dict[str, Any]) -> dict[str, Any]:
    return {decision["key"]: decision["default"] for decision in preflight.get("decisions", [])}


def _render_preflight_contents(preflight: dict[str, Any], disabled: bool = False) -> dict[str, Any]:
    if preflight.get("detected_frequency"):
        st.caption(f"Selected-series frequency: **{preflight['detected_frequency']}**")

    for message in preflight.get("issues", []):
        st.info(message)
    for message in preflight.get("warnings", []):
        st.warning(message)

    choices = dict(st.session_state.get("_preflight_options_current") or _preflight_defaults(preflight))
    for decision in preflight.get("decisions", []):
        key = decision["key"]
        options = decision["options"]
        default = choices.get(key, decision["default"])
        default_index = options.index(default) if default in options else 0
        choices[key] = st.selectbox(
            decision["label"],
            options=options,
            index=default_index,
            help=decision["message"],
            disabled=disabled,
            key=f"preflight_choice_{key}",
        )
    return choices


def _render_reasoning(steps: list[dict[str, Any]]) -> None:
    """Helper to render agent reasoning traces in an expander."""
    if not steps:
        st.info("No detailed reasoning trace captured for this step.")
        return
    
    for i, step in enumerate(steps):
        with st.container():
            st.markdown(f"**Step {i+1}**")
            # Capture the thought/log. If 'thought' is missing, try 'thought_log' or 'log'
            thought = (step.get("thought") or step.get("log") or "").strip()
            
            # Remove the "Thought:" prefix if the LLM included it in the log
            if thought.lower().startswith("thought:"):
                thought = thought[8:].strip()
            
            if thought:
                st.caption(thought)
                
            if step.get("observation"):
                st.info(f"Observation: {step['observation']}")


def _render_preflight_dialog(preflight: dict[str, Any], disabled: bool = False) -> bool:
    choices = _render_preflight_contents(preflight, disabled=disabled)
    if st.button("Apply Preflight Choices", disabled=disabled, use_container_width=True):
        st.session_state._preflight_options_current = choices
        st.rerun()
    return False


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
                    st.error(f"🌐 Backend Connection Error: {exc}. Verify BACKEND_URL and service names in docker-compose.")

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

    st.markdown("---")
    show_advanced = st.toggle("🕵️ Advanced Mode", value=False, help="Show AI reasoning traces and internal thoughts for each step.")

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
            resp = requests.post(
                f"{BACKEND_URL}/preflight",
                json={
                    "file_id": info["file_id"],
                    "forecast_horizon": forecast_horizon,
                    "date_col": date_col,
                    "value_col": value_col,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                preflight = resp.json()
                decisions = preflight.get("decisions", [])
                saved_options = st.session_state.get("_preflight_options_current")
                preflight_options = saved_options or _preflight_defaults(preflight)

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
                            choices = _render_preflight_contents(preflight, disabled=is_running)
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
            resp = requests.post(
                f"{BACKEND_URL}/analyze",
                json={
                    "file_id": info["file_id"],
                    "forecast_horizon": forecast_horizon,
                    "date_col": date_col,
                    "value_col": value_col,
                    "forced_model": forced_model,
                    "user_prompt": st.session_state.get("_user_prompt"),
                    "preflight_options": st.session_state.get("_preflight_options"),
                },
                timeout=30,
            )
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
            resp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=10)
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
    tab_report, tab_forecast, tab_model, tab_stats, tab_quality, tab_trace, tab_overview = st.tabs([
        "📄 Report",
        "🔮 Forecast",
        "🤖 Model Selection",
        "📐 Statistical Analysis",
        "🔍 Data Quality",
        "🕵️ AI Reasoning Trace",
        "📊 Data Overview",
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
            fig = go.Figure(result["chart_historical"], skip_invalid=True)
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

        if show_advanced:
            with st.expander("🕵️ View Data Validation Reasoning", expanded=False):
                _render_reasoning(v.get("reasoning_steps", []))

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

        if show_advanced:
            with st.expander("🕵️ View Statistical Analysis Reasoning", expanded=False):
                _render_reasoning(s.get("reasoning_steps", []))

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

    # ── Tab 4: Model Selection ────────────────────────────────────────────────
    with tab_model:
        m = result["model_selection"]
        st.success(f"**Selected Model: {m['selected_model']}**")
        st.markdown("**Rationale:**")
        st.write(m["explanation"])

        if show_advanced:
            with st.expander("🕵️ View Model Selection Reasoning", expanded=False):
                _render_reasoning(m.get("reasoning_steps", []))

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

    # ── Tab 5: Forecast ───────────────────────────────────────────────────────
    with tab_forecast:
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

    # ── Tab 6: Report ─────────────────────────────────────────────────────────
    with tab_report:
        report_text = result.get("report", "Report not available.")
        try:
            pdf_bytes = _report_to_pdf(report_text)
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

    # ── Tab 7: AI Trace ───────────────────────────────────────────────────────
    with tab_trace:
        st.subheader("🕵️ Full AI Reasoning Trace")
        st.write("This log shows the internal 'monologue' and tool usage of the AI agents as they processed your request.")
        
        with st.expander("1. Data Validation Agent", expanded=True):
            _render_reasoning(result["validation"].get("reasoning_steps", []))
            
        with st.expander("2. Statistical Analysis Agent", expanded=True):
            _render_reasoning(result["statistical"].get("reasoning_steps", []))
            
        with st.expander("3. Model Selection Agent", expanded=True):
            _render_reasoning(result["model_selection"].get("reasoning_steps", []))
            
        with st.expander("4. Forecasting Agent", expanded=True):
            _render_reasoning(result["forecast"].get("reasoning_steps", []))
            
        with st.expander("5. Report Generation Agent", expanded=True):
            _render_reasoning(result.get("report_reasoning", []))
            
        st.info("💡 Advanced Mode (toggle in sidebar) also shows these traces inline within each specific analysis tab.")

else:
    st.info("Upload a time series file and click **Run Analysis** to get started.")
