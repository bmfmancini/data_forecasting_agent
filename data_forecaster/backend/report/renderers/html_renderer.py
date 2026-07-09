"""HTML renderer for the :class:`ExecutiveReport`.

Produces clean, sanitised HTML suitable for frontend injection or API
responses.  Dashboard items are rendered as cards; health indicators and
prediction intervals as styled tables; recommendations as structured blocks
with evidence references.

The LLM never generates this HTML — the renderer assembles it from the
pre-computed structured fields and LLM-generated narrative strings.
"""

from __future__ import annotations

from html import escape

from report.models import ExecutiveReport
from report.rules import DASHBOARD_STATUS_COLORS


class HTMLRenderer:
    """Render an :class:`ExecutiveReport` to HTML."""

    def render(self, report: ExecutiveReport) -> str:
        """Produce the full HTML report string.

        Args:
            report: Populated :class:`ExecutiveReport` with narratives.

        Returns:
            HTML string with dashboard cards, tables, and narrative sections.
        """
        # The frontend expects a single report body to parse into segments.
        # The dashboard, health indicators, etc., are now rendered by the
        # frontend template directly from the structured `executive_report` model.
        # This renderer should only produce the core narrative content.
        return (
            f"{self._render_executive_summary(report)}"
            f"{self._render_data_quality(report)}"
            f"{self._render_historical_analysis(report)}"
            f"{self._render_forecast_outlook(report)}"
            f"{self._render_model_comparison(report)}"
            f"{self._render_explainability(report)}"
            f"{self._render_statistical_audit(report)}"
        )

    # ── Dashboard Cards ───────────────────────────────────────────────────

    def _render_dashboard(self, report: ExecutiveReport) -> str:
        """Render dashboard items as Bootstrap-style cards."""
        cards: list[str] = []
        for item in report.dashboard.widgets:
            color = DASHBOARD_STATUS_COLORS.get(item.status, "primary")
            cards.append(
                f'<div class="card dashboard-card border-{color} mb-2">'
                f'<div class="card-body">'
                f'<h6 class="card-title">{item.icon} {escape(item.title)}</h6>'
                f'<p class="h4 mb-1">{escape(item.value)}</p>'
                f'<small class="text-muted">{escape(item.description)}</small>'
                f"</div></div>"
            )
        grid = (
            '<div class="row row-cols-1 row-cols-md-3 g-3">'
            + "".join(
                f'<div class="col">{card}</div>' for card in cards
            )
            + "</div>"
        )
        return f'<section class="report-dashboard">{grid}</section>'

    # ── Confidence Badge ──────────────────────────────────────────────────

    def _render_confidence(self, report: ExecutiveReport) -> str:
        """Render the confidence score as a badge with explanation."""
        c = report.confidence
        color = self._label_color(c.label)
        return (
            f'<section class="report-confidence mb-3">'
            f'<span class="badge bg-{color} fs-6">'
            f"Forecast Confidence: {c.score}/100 — {escape(c.label)}"
            f"</span>"
            f"<p class=\"mt-2 text-muted\">{escape(c.explanation)}</p>"
            f"</section>"
        )

    # ── Health Indicators Table ───────────────────────────────────────────

    def _render_health_indicators(self, report: ExecutiveReport) -> str:
        """Render health indicators as an HTML table."""
        rows = "".join(
            f"<tr><td>{escape(hi.indicator)}</td>"
            f"<td>{escape(hi.status)}</td>"
            f"<td>{escape(hi.detail)}</td></tr>"
            for hi in report.health_indicators
        )
        return (
            '<section class="report-health mb-3">'
            "<h5>Forecast Health Indicators</h5>"
            '<table class="table table-sm table-striped">'
            "<thead><tr><th>Indicator</th><th>Status</th>"
            "<th>Detail</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</section>"
        )

    # ── Executive Summary ─────────────────────────────────────────────────

    def _render_executive_summary(self, report: ExecutiveReport) -> str:
        """Render the executive summary section."""
        s = report.executive_summary
        narrative = (
            f"<p>{escape(s.narrative)}</p>" if s.narrative else ""
        )
        return (
            '<section class="report-summary mb-3">'
            "<h5>Executive Summary</h5>"
            f"<p><strong>Strategic Outlook:</strong> {escape(s.strategic_outlook)}</p>"
            f"<p><strong>Expected Growth:</strong> {escape(s.expected_growth)}</p>"
            f"<p><strong>Confidence Level:</strong> {escape(s.confidence_level)}</p>"
            f"<p><strong>Primary Risk:</strong> {escape(s.primary_risk)}</p>"
            f"<p><strong>Recommended Action:</strong> {escape(s.recommended_action)}</p>"
            f"{narrative}"
            "</section>"
        )

    # ── Data Quality ──────────────────────────────────────────────────────

    def _render_data_quality(self, report: ExecutiveReport) -> str:
        """Render the data quality section."""
        dq = report.data_quality
        color = self._rating_color(dq.rating)
        issues = (
            "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in dq.issues) + "</ul>"
            if dq.issues else ""
        )
        narrative = f"<p>{escape(dq.narrative)}</p>" if dq.narrative else ""
        return (
            '<section class="report-data-quality mb-3">'
            "<h5>Data Quality Summary</h5>"
            f'<span class="badge bg-{color}">{escape(dq.rating)}</span>'
            f"<p class='mt-2'>{escape(dq.rating_explanation)}</p>"
            f"{narrative}"
            f"<p>Completeness: {dq.completeness_pct:.1f}% | "
            f"Missing: {dq.missing_values} | Duplicates: {dq.duplicate_timestamps} | "
            f"Gaps: {dq.missing_timestamps}</p>"
            f"{issues}"
            "</section>"
        )
        
    # ── Historical Analysis ───────────────────────────────────────────────

    def _render_historical_analysis(self, report: ExecutiveReport) -> str:
        """Render the historical analysis section."""
        h = report.historical_analysis
        narrative = f"<p>{escape(h.narrative)}</p>" if h.narrative else ""
        return (
            '<section class="report-historical-analysis mb-3">'
            "<h5>Historical Performance & Trend Analysis</h5>"
            f"<p><strong>Trend Direction:</strong> {escape(h.trend_direction)}</p>"
            f"<p><strong>Seasonal Pattern:** {'Every ' + str(h.seasonal_period) + ' periods' if h.seasonal_period else 'None detected'}</p>"
            f"{narrative}"
            '<p class="mt-3"><strong>Figure: Historical Data & Decomposition</strong></p>'
            "<p>[VISUAL:HISTORICAL]</p>"
            "<p>[VISUAL:STL]</p>"
            f"{narrative}"
            "</section>"
        )

    # ── Forecast Outlook ──────────────────────────────────────────────────

    def _render_forecast_outlook(self, report: ExecutiveReport) -> str:
        """Render the forecast outlook section."""
        f = report.forecast_outlook
        m = f.metrics
        narrative = f"<p>{escape(f.narrative)}</p>" if f.narrative else ""
        return (
            '<section class="report-forecast-outlook mb-3">'
            "<h5>Future Growth & Forecast Outlook</h5>"
            f"<p><strong>Projected Change:</strong> {m.pct_change:+.1f}% over {m.horizon} periods.</p>"
            f"{narrative}"
            '<p class="mt-3"><strong>Figure: Forecast with Prediction Intervals</strong></p>'
            "<p>[VISUAL:FORECAST]</p>"
            f"{narrative}"
            "</section>"
        )

    # ── Model Comparison ──────────────────────────────────────────────────

    def _render_model_comparison(self, report: ExecutiveReport) -> str:
        """Render the model comparison table."""
        mc = report.model_comparison
        rows = "".join(
            f"<tr><td>{escape(e.model)}</td>"
            f"<td>{e.rmse}</td>"
            f"<td>{e.mae}</td>"
            f"<td>{e.mape}%</td>"
            f"<td>{'✓' if e.selected else ''}</td></tr>"
            for e in mc.entries
        )
        narrative = f"<p>{escape(mc.narrative)}</p>" if mc.narrative else ""
        return (
            '<section class="report-model-comparison mb-3">'
            "<h5>Forecasting Approach & Model Comparison</h5>"
            f"<p><strong>Selected Model:</strong> {escape(mc.selected_model)}</p>"
            f"<p class='small text-muted'>{escape(mc.selection_rationale)}</p>"
            f"{narrative}"
            '<table class="table table-sm table-striped">'
            "<thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Selected</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            '<p class="mt-3"><strong>Figure: Model Diagnostics & Comparison</strong></p>'
            "<p>[VISUAL:ACF_PACF]</p>"
            "<p>[VISUAL:COMPARISON]</p>"
            "</section>"
        )

    # ── Explainability ────────────────────────────────────────────────────

    def _render_explainability(self, report: ExecutiveReport) -> str:
        """Render the explainability section."""
        ex = report.explainability
        narrative = f"<p>{escape(ex.narrative)}</p>" if ex.narrative else ""
        items = "".join(
            f"<li><strong>{escape(item.finding)}:</strong> {escape(item.interpretation)} "
            f"<em class='small text-muted'>(Evidence: {escape(item.evidence)})</em></li>"
            for item in ex.findings
        )
        return (
            '<section class="report-explainability mb-3">'
            "<h5>Explainability — Why These Conclusions</h5>"
            f"{narrative}"
            f"<ul>{items}</ul>"
            "</section>"
        )

    # ── Statistical Audit ─────────────────────────────────────────────────

    def _render_statistical_audit(self, report: ExecutiveReport) -> str:
        """Render the statistical audit section."""
        sa = report.statistical_audit
        narrative = f"<p>{escape(sa.narrative)}</p>" if sa.narrative else ""
        concerns = "".join(f"<li>{escape(c)}</li>" for c in sa.key_concerns)
        evidence = "".join(f"<li>{escape(e)}</li>" for e in sa.strongest_evidence)
        return (
            '<section class="report-statistical-audit mb-3">'
            "<h5>Statistical Audit Summary</h5>"
            f"<p><strong>Verdict:</strong> {escape(sa.verdict.upper())}</p>"
            f"{narrative}"
            f"{'<h6>Key Concerns</h6><ul>' + concerns + '</ul>' if concerns else ''}"
            f"{'<h6>Strongest Evidence</h6><ul>' + evidence + '</ul>' if evidence else ''}"
            "</section>"
        )

    # ── Risks ─────────────────────────────────────────────────────────────

    def _render_risks(self, report: ExecutiveReport) -> str:
        """Render the risks section."""
        if not report.risks:
            return ""
        blocks: list[str] = []
        for risk in report.risks:
            evidence_items = "".join(f"<li>{escape(ev)}</li>" for ev in risk.evidence)
            evidence = f'<ul class="small text-muted">{evidence_items}</ul>' if evidence_items else ""
            blocks.append(
                f'<div class="risk-block mb-2">'
                f"<h6>{escape(risk.category)} — {escape(risk.severity)}</h6>"
                f"<p><strong>Risk:</strong> {escape(risk.description)}</p>"
                f"<p class='small'><strong>Potential Impact:</strong> {escape(risk.potential_impact)}</p>"
                f"<p class='small'><strong>Mitigation:</strong> {escape(risk.mitigation)}</p>"
                f"{evidence}"
                f"</div>"
            )
        return (
            '<section class="report-risks mb-3">'
            "<h5>Strategic Risks & Operational Constraints</h5>"
            + "".join(blocks)
            + "</section>"
        )

    # ── Assumptions ───────────────────────────────────────────────────────

    def _render_assumptions(self, report: ExecutiveReport) -> str:
        """Render the assumptions section."""
        if not report.assumptions:
            return ""
        items = "".join(
            f"<li><strong>{escape(a.assumption)}</strong>"
            f"<p class='small text-muted'><em>Consequence if false: {escape(a.consequence_if_false)}</em></p></li>"
            for a in report.assumptions
        )
        return (
            '<section class="report-assumptions mb-3">'
            "<h5>Critical Business Assumptions</h5>"
            f"<ul>{items}</ul>"
            "</section>"
        )

    # ── Prediction Intervals Table ────────────────────────────────────────

    def _render_prediction_intervals(self, report: ExecutiveReport) -> str:
        """Render prediction intervals as an HTML table."""
        intervals = report.forecast_outlook.metrics.prediction_intervals
        if not intervals:
            return ""
        rows = "".join(
            f"<tr><td>{escape(pi.date)}</td>"
            f"<td>{pi.forecast}</td>"
            f"<td>{pi.lower_ci}</td>"
            f"<td>{pi.upper_ci}</td></tr>"
            for pi in intervals
        )
        return (
            '<section class="report-prediction-intervals mb-3">'
            f"<h5>Prediction Intervals ({escape(intervals.confidence_level)})</h5>"
            '<table class="table table-sm table-striped">'
            "<thead><tr><th>Date</th><th>Forecast</th>"
            "<th>Lower Bound</th><th>Upper Bound</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</section>"
        )

    # ── Recommendations ───────────────────────────────────────────────────

    def _render_recommendations(self, report: ExecutiveReport) -> str:
        """Render recommendations as structured blocks with evidence."""
        if not report.recommendations:
            return ""
        blocks: list[str] = []
        for rec in report.recommendations:
            color = self._priority_color(rec.priority)
            evidence_items = "".join(
                f"<li>{escape(ev.metric)}: {escape(ev.value)} "
                f"(from {escape(ev.source_section)})</li>"
                for ev in rec.supporting_evidence
            )
            evidence = (
                f'<ul class="small text-muted">{evidence_items}</ul>'
                if evidence_items else ""
            )
            text = rec.narrative if rec.narrative else rec.recommendation
            blocks.append(
                f'<div class="recommendation-block mb-2">'
                f'<span class="badge bg-{color}">{escape(rec.priority)}</span> '
                f"<strong>{escape(text)}</strong>"
                f"<p class='small'>{escape(rec.rationale)}</p>"
                f"<p class='small'><em>Expected outcome: "
                f"{escape(rec.expected_outcome)}</em></p>"
                f"{evidence}"
                f"</div>"
            )
        return (
            '<section class="report-recommendations mb-3">'
            "<h5>Executive Recommendations</h5>"
            + "".join(blocks)
            + "</section>"
        )

    # ── Metadata ──────────────────────────────────────────────────────────

    @staticmethod
    def _label_color(label: str) -> str:
        """Return a Bootstrap colour class for a confidence label."""
        if label == "High":
            return "success"
        if label == "Medium":
            return "warning"
        return "danger"

    @staticmethod
    def _rating_color(rating: str) -> str:
        """Return a Bootstrap colour class for a data quality rating."""
        if rating == "Good":
            return "success"
        if rating == "Fair":
            return "warning"
        return "danger"

    @staticmethod
    def _priority_color(priority: str) -> str:
        """Return a Bootstrap colour class for a recommendation priority."""
        if priority == "High":
            return "danger"
        if priority == "Medium":
            return "warning"
        return "info"

    def _render_metadata(self, report: ExecutiveReport) -> str:
        """Render report metadata as a small table."""
        m = report.metadata
        return (
            '<section class="report-metadata mt-4">'
            "<h6>Report Metadata</h6>"
            '<table class="table table-sm table-bordered small">'
            f"<tr><td>Engine Version</td><td>{escape(m.engine_version)}</td></tr>"
            f"<tr><td>Generated At</td><td>{escape(m.generated_at)}</td></tr>"
            f"<tr><td>Forecast Horizon</td><td>{m.forecast_horizon} periods</td></tr>"
            f"<tr><td>Selected Model</td><td>{escape(m.selected_model)}</td></tr>"
            f"<tr><td>Dataset Frequency</td><td>{escape(m.dataset_frequency)}</td></tr>"
            f"<tr><td>Data Quality</td><td>{escape(m.data_quality_rating)}</td></tr>"
            f"<tr><td>Row Count</td><td>{m.row_count}</td></tr>"
            "</table>"
            "</section>"
        )