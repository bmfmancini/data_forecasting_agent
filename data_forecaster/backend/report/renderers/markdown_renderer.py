"""Markdown renderer for the :class:`ExecutiveReport`.

Iterates the structured report model section-by-section and produces
Markdown text with 12 sections + appendix.  Visual tags (``[VISUAL:TAG]``)
are emitted by the renderer in their designated sections so the frontend
can still parse them for chart embedding.

The LLM never generates this Markdown — the renderer assembles it from
the pre-computed structured fields and LLM-generated narrative strings.
"""

from __future__ import annotations

from report.models import ExecutiveReport, format_metric


def _sanitize_cell(value: str) -> str:
    """Escape characters that would break a Markdown table cell.

    Replaces pipe characters and newlines so free-text values (e.g. LLM-
    derived rejection reasons) render safely inside pipe-delimited tables.

    Args:
        value: Raw cell text.

    Returns:
        Sanitised cell text safe for Markdown table insertion.
    """
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


class MarkdownRenderer:
    """Render an :class:`ExecutiveReport` to Markdown text."""

    def render(self, report: ExecutiveReport) -> str:
        """Produce the full Markdown report string.

        Args:
            report: Populated :class:`ExecutiveReport` with narratives.

        Returns:
            Markdown string with 12 sections + appendix.
        """
        sections: list[str] = []
        sections.append(self._render_dashboard(report))
        sections.append(self._render_executive_summary(report))
        sections.append(self._render_data_quality(report))
        sections.append(self._render_historical_analysis(report))
        sections.append(self._render_forecast_outlook(report))
        sections.append(self._render_model_comparison(report))
        sections.append(self._render_reliability(report))
        sections.append(self._render_explainability(report))
        sections.append(self._render_statistical_audit(report))
        sections.append(self._render_risks(report))
        sections.append(self._render_recommendations(report))
        sections.append(self._render_assumptions(report))
        sections.append(self._render_appendix(report))
        return "\n\n---\n\n".join(sections)

    # ── Section 1: Executive Dashboard ────────────────────────────────────

    def _render_dashboard(self, report: ExecutiveReport) -> str:
        """Render the dashboard as a Markdown table."""
        lines = ["## 1. Executive Dashboard", ""]
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        for item in report.dashboard.widgets:
            lines.append(f"| {item.icon} {item.title} | {item.value} | {item.status} |")
        lines.append("")
        return "\n".join(lines)

    # ── Section 2: Executive Summary ──────────────────────────────────────

    def _render_executive_summary(self, report: ExecutiveReport) -> str:
        """Render the executive summary with structured fields + narrative."""
        s = report.executive_summary
        lines = ["## 2. Executive Summary", ""]
        lines.append(f"**Strategic Outlook:** {s.strategic_outlook}")
        lines.append("")
        lines.append(f"**Forecast Endpoint Change:** {s.expected_growth}")
        lines.append("")
        lines.append(f"**Confidence Level:** {s.confidence_level}")
        lines.append("")
        lines.append(f"**Primary Risk:** {s.primary_risk}")
        lines.append("")
        lines.append(f"**Recommended Action:** {s.recommended_action}")
        if s.narrative:
            lines.append("")
            lines.append(s.narrative)
        return "\n".join(lines)

    # ── Section 3: Data Quality Summary ───────────────────────────────────

    def _render_data_quality(self, report: ExecutiveReport) -> str:
        """Render the data quality summary."""
        dq = report.data_quality
        lines = ["## 3. Data Quality Summary", ""]
        lines.append(f"**Overall Rating:** {dq.rating}")
        lines.append("")
        lines.append(dq.rating_explanation)
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Missing Values | {dq.missing_values} |")
        lines.append(f"| Duplicate Timestamps | {dq.duplicate_timestamps} |")
        lines.append(f"| Missing Timestamps (Gaps) | {dq.missing_timestamps} |")
        lines.append(f"| Outlier Count | {dq.outlier_count} |")
        lines.append(f"| Outlier Ratio | {dq.outlier_ratio:.1%} |")
        lines.append(f"| Regular Intervals | {'Yes' if dq.is_regular else 'No'} |")
        lines.append(f"| Frequency | {dq.frequency} |")
        lines.append(f"| Completeness | {dq.completeness_pct:.1f}% |")
        if dq.issues:
            lines.append("")
            lines.append("**Issues Identified:**")
            for issue in dq.issues:
                lines.append(f"- {issue}")
        if dq.narrative:
            lines.append("")
            lines.append(dq.narrative)
        return "\n".join(lines)

    # ── Section 4: Historical Performance ─────────────────────────────────

    def _render_historical_analysis(self, report: ExecutiveReport) -> str:
        """Render historical analysis with visual tags."""
        h = report.historical_analysis
        lines = ["## 4. Historical Performance & Trend Analysis", ""]
        lines.append(f"**Trend Direction:** {h.trend_direction}")
        lines.append(f"**Trend Detected:** {'Yes' if h.has_trend else 'No'}")
        if h.seasonal_period:
            lines.append(f"**Seasonal Pattern:** Every {h.seasonal_period} periods")
        else:
            lines.append("**Seasonal Pattern:** None detected")
        lines.append(
            f"**Statistical Stability:** {'Stable' if h.is_stationary else 'Changing over time'}"
        )
        if h.narrative:
            lines.append("")
            lines.append(h.narrative)
        lines.append("")
        lines.append("**Figure: Historical Data**")
        lines.append(
            "The historical time series showing the underlying trend and seasonal patterns."
        )
        lines.append("")
        lines.append("[VISUAL:HISTORICAL]")
        lines.append("")
        lines.append("[VISUAL:STL]")
        return "\n".join(lines)

    # ── Section 5: Forecast Outlook ───────────────────────────────────────

    def _render_forecast_outlook(self, report: ExecutiveReport) -> str:
        """Render forecast outlook with prediction intervals table."""
        m = report.forecast_outlook.metrics
        lines = ["## 5. Future Growth & Forecast Outlook", ""]
        lines.append(
            f"**Forecast Horizon:** {m.horizon} periods ({m.first_date} → {m.last_date})"
        )
        lines.append(f"**Endpoint Change:** {m.pct_change:+.1f}%")
        lines.append(f"**Endpoint Direction:** {m.endpoint_direction}")
        lines.append(f"**Forecast Pattern:** {m.forecast_pattern}")
        if m.peak_value is not None:
            peak_context = f" on {m.peak_date}" if m.peak_date else ""
            lines.append(
                f"**Seasonal Peak:** {m.peak_value}{peak_context} "
                f"({format_metric(m.peak_change_pct, '+.1f')}% versus the first forecast period)"
            )
        lines.append(f"**Start Value:** {m.first_value}")
        lines.append(f"**End Value:** {m.last_value}")
        lines.append(
            "**Metric Provenance:** Model selection used rolling-origin metrics; "
            "the untouched final-test metrics below were not used for ranking."
        )
        final_rmse = m.final_test_metrics.get("rmse")
        final_mae = m.final_test_metrics.get("mae")
        lines.append(
            f"**Untouched Final Test:** RMSE {format_metric(final_rmse)}, "
            f"MAE {format_metric(final_mae)}"
        )
        if m.final_test_assessment:
            lines.append(f"**Recent Holdout Assessment:** {m.final_test_assessment}")
        if report.forecast_outlook.narrative:
            lines.append("")
            lines.append(report.forecast_outlook.narrative)
        lines.append("")
        interval_label = m.interval_label
        if not m.prediction_intervals:
            lines.append("### Prediction Intervals Unavailable")
            lines.append("")
            lines.append(
                "The forecasting model did not produce usable prediction-interval "
                "bounds; no 95% planning range is shown."
            )
            lines.append("")
            lines.append("**Figure: Point Forecast**")
            lines.append("")
            lines.append("[VISUAL:FORECAST]")
            return "\n".join(lines)
        interval_heading = (
            "Estimated 95% Prediction Intervals (coverage not evaluated)"
            if interval_label == "experimental"
            else "Model-Based 95% Prediction Intervals"
        )
        lines.append(f"### {interval_heading}")
        lines.append("")
        lines.append("| Date | Forecast | Lower Bound | Upper Bound |")
        lines.append("|------|----------|-------------|-------------|")
        for pi in m.prediction_intervals:
            lines.append(
                f"| {pi.date} | {pi.forecast} | {pi.lower_ci} | {pi.upper_ci} |"
            )
        lines.append("")
        figure_label = (
            "Forecast with Estimated Prediction Intervals"
            if interval_label == "experimental"
            else "Forecast with Model-Based Prediction Intervals"
        )
        lines.append(f"**Figure: {figure_label}**")
        lines.append(
            "The projected values with an estimated 95% planning range; empirical "
            "coverage was not evaluated."
            if interval_label == "experimental"
            else "The projected values with a model-based 95% planning range."
        )
        lines.append("")
        lines.append("[VISUAL:FORECAST]")
        return "\n".join(lines)

    # ── Section 6: Forecasting Approach ───────────────────────────────────

    def _render_model_comparison(self, report: ExecutiveReport) -> str:
        """Render model comparison with metrics table."""
        mc = report.model_comparison
        lines = ["## 6. Forecasting Approach & Model Comparison", ""]
        lines.append(f"**Selected Model:** {mc.selected_model}")
        lines.append("")
        if mc.narrative:
            lines.append(mc.narrative)
            lines.append("")
        lines.append(
            "| Model | RMSE | MAE | MAPE | WAPE | MASE | Selected | Rejected Reason |"
        )
        lines.append(
            "|-------|------|-----|------|------|------|----------|-----------------|"
        )
        for entry in mc.entries:
            selected = "✓" if entry.selected else ""
            rejected = _sanitize_cell(entry.rejected_reason or "")
            lines.append(
                f"| {entry.model} | {format_metric(entry.rmse)} | "
                f"{format_metric(entry.mae)} | "
                f"{format_metric(entry.mape, '.2f')}% | "
                f"{format_metric(entry.wape, '.2f')}% | "
                f"{format_metric(entry.mase)} | {selected} | {rejected} |"
            )
        lines.append("")
        lines.append("[VISUAL:ACF_PACF]")
        return "\n".join(lines)

    # ── Section 7: Forecast Reliability ───────────────────────────────────

    def _render_reliability(self, report: ExecutiveReport) -> str:
        """Render reliability with health indicators table + confidence."""
        c = report.confidence
        lines = ["## 7. Forecast Reliability & Performance Assessment", ""]
        lines.append(f"**Confidence Score:** {c.score}/100 — {c.label}")
        lines.append("")
        lines.append(c.explanation)
        lines.append("")
        lines.append("### Forecast Health Indicators")
        lines.append("")
        lines.append("| Indicator | Status | Detail |")
        lines.append("|-----------|--------|--------|")
        for hi in report.health_indicators:
            detail = _sanitize_cell(hi.detail)
            lines.append(f"| {hi.indicator} | {hi.status} | {detail} |")
        lines.append("")
        lines.append("**Contributing Factors:**")
        for factor in c.contributing_factors:
            lines.append(f"- {factor}")
        lines.append("")
        lines.append("[VISUAL:COMPARISON]")
        return "\n".join(lines)

    # ── Section 8: Explainability ─────────────────────────────────────────

    def _render_explainability(self, report: ExecutiveReport) -> str:
        """Render the explainability section."""
        e = report.explainability
        lines = ["## 8. Explainability — Why These Conclusions", ""]
        if e.narrative:
            lines.append(e.narrative)
            lines.append("")
        for item in e.findings:
            lines.append(f"- **{item.finding}** — {item.interpretation}")
            lines.append(f"  *Evidence: {item.evidence}*")
        return "\n".join(lines)

    # ── Section 9: Statistical Audit Summary ──────────────────────────────

    def _render_statistical_audit(self, report: ExecutiveReport) -> str:
        """Render the statistical audit summary."""
        a = report.statistical_audit
        lines = ["## 9. Statistical Audit Summary", ""]
        lines.append(f"**Verdict:** {a.verdict.upper()}")
        if a.narrative:
            lines.append("")
            lines.append(a.narrative)
        if a.strongest_evidence:
            lines.append("")
            lines.append("**Strongest Evidence:**")
            for ev in a.strongest_evidence:
                lines.append(f"- {ev}")
        if a.key_concerns:
            lines.append("")
            lines.append("**Key Concerns:**")
            for concern in a.key_concerns:
                lines.append(f"- {concern}")
        if a.recommended_follow_up:
            lines.append("")
            lines.append("**Recommended Follow-up:**")
            for fu in a.recommended_follow_up:
                lines.append(f"- {fu}")
        return "\n".join(lines)

    # ── Section 10: Strategic Risks ───────────────────────────────────────

    def _render_risks(self, report: ExecutiveReport) -> str:
        """Render the strategic risks section."""
        lines = ["## 10. Strategic Risks & Operational Constraints", ""]
        if not report.risks:
            lines.append("No significant risks were identified from the analysis.")
            return "\n".join(lines)
        for risk in report.risks:
            lines.append(f"### {risk.category} — {risk.severity}")
            lines.append(f"**Risk:** {risk.description}")
            lines.append(f"**Potential Impact:** {risk.potential_impact}")
            lines.append(f"**Mitigation:** {risk.mitigation}")
            if risk.evidence:
                lines.append("**Evidence:**")
                for ev in risk.evidence:
                    lines.append(f"- {ev}")
            lines.append("")
        return "\n".join(lines)

    # ── Section 11: Executive Recommendations ─────────────────────────────

    def _render_recommendations(self, report: ExecutiveReport) -> str:
        """Render recommendations with evidence refs."""
        lines = ["## 11. Executive Recommendations & Next Steps", ""]
        if not report.recommendations:
            lines.append("No specific recommendations at this time.")
            return "\n".join(lines)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"### {i}. [{rec.priority}] {rec.recommendation}")
            text = rec.narrative if rec.narrative else rec.recommendation
            lines.append(f"**Action:** {text}")
            lines.append(f"**Rationale:** {rec.rationale}")
            lines.append(f"**Expected Outcome:** {rec.expected_outcome}")
            if rec.supporting_evidence:
                lines.append("**Supporting Evidence:**")
                for ev in rec.supporting_evidence:
                    lines.append(
                        f"- {ev.metric}: {ev.value} (from {ev.source_section})"
                    )
            lines.append("")
        return "\n".join(lines)

    # ── Section 12: Critical Business Assumptions ─────────────────────────

    def _render_assumptions(self, report: ExecutiveReport) -> str:
        """Render the critical business assumptions."""
        lines = ["## 12. Critical Business Assumptions", ""]
        if not report.assumptions:
            lines.append("No specific assumptions identified.")
            return "\n".join(lines)
        for i, assumption in enumerate(report.assumptions, 1):
            lines.append(f"{i}. **{assumption.assumption}**")
            lines.append(
                f"   *Consequence if false:* {assumption.consequence_if_false}"
            )
            lines.append("")
        return "\n".join(lines)

    # ── Appendix ──────────────────────────────────────────────────────────

    def _render_appendix(self, report: ExecutiveReport) -> str:
        """Render the appendix with metadata and raw metrics."""
        meta = report.metadata
        lines = ["## Appendix — Report Metadata", ""]
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Engine Version | {meta.engine_version} |")
        lines.append(f"| Generated At | {meta.generated_at} |")
        lines.append(f"| Forecast Horizon | {meta.forecast_horizon} periods |")
        lines.append(f"| Models Evaluated | {', '.join(meta.models_evaluated)} |")
        lines.append(f"| Selected Model | {meta.selected_model} |")
        lines.append(f"| Dataset Frequency | {meta.dataset_frequency} |")
        lines.append(f"| Data Quality Rating | {meta.data_quality_rating} |")
        lines.append(f"| Row Count | {meta.row_count} |")
        return "\n".join(lines)
