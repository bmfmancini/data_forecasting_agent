from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.logging_config import get_logger
from schemas import ForecastResult

logger = get_logger(__name__)


def plot_historical(series: pd.Series) -> dict[str, Any]:
    """Line chart of the historical time series."""
    dates = _index_to_str(series)
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=series.values.tolist(),
            mode="lines",
            name="Historical",
            line=dict(color="#2563EB", width=2),
        )
    )
    fig.update_layout(
        title="Historical Time Series",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
    )
    return _fig_to_dict(fig)


def plot_stl(
    series: pd.Series, stl_data: dict[str, list[float]], seasonal_period: int
) -> dict[str, Any]:
    """4-panel STL decomposition chart: observed, trend, seasonal, residual."""
    dates = _index_to_str(series)
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    def _trace(y: list, name: str, color: str) -> go.Scatter:
        return go.Scatter(
            x=dates, y=y, mode="lines", name=name, line=dict(color=color, width=1.5)
        )

    fig.add_trace(_trace(series.values.tolist(), "Observed", "#2563EB"), row=1, col=1)
    fig.add_trace(_trace(stl_data["trend"], "Trend", "#16A34A"), row=2, col=1)
    fig.add_trace(_trace(stl_data["seasonal"], "Seasonal", "#D97706"), row=3, col=1)
    fig.add_trace(_trace(stl_data["residual"], "Residual", "#DC2626"), row=4, col=1)

    fig.update_layout(
        title=f"STL Decomposition (period={seasonal_period})",
        height=700,
        template="plotly_white",
        showlegend=False,
    )
    return _fig_to_dict(fig)


def plot_acf_pacf(acf_values: list, pacf_values: list, lags: list) -> str:
    """ACF and PACF bar chart — returns base64-encoded PNG."""
    n = len(acf_values)
    conf = 1.96 / np.sqrt(max(n, 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    for ax, values, title, ylabel in [
        (ax1, acf_values, "AutoCorrelation Function (ACF)", "ACF"),
        (ax2, pacf_values, "Partial AutoCorrelation Function (PACF)", "PACF"),
    ]:
        ax.bar(lags, values, width=0.3, color="#2563EB", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(
            conf, color="red", linestyle="--", linewidth=0.9, alpha=0.7, label="95% CI"
        )
        ax.axhline(-conf, color="red", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Lag")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    logger.debug("ACF/PACF PNG generated (%d bytes base64)", len(img_b64))
    return img_b64


def plot_forecast(series: pd.Series, forecast_result: ForecastResult) -> dict[str, Any]:
    """Historical series + forecast line + 95% CI ribbon."""
    hist_dates = _index_to_str(series)
    fc_dates = forecast_result.forecast_dates or [
        str(i) for i in range(len(forecast_result.forecast))
    ]

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=series.values.tolist(),
            mode="lines",
            name="Historical",
            line=dict(color="#2563EB", width=2),
        )
    )

    # Confidence interval ribbon
    fig.add_trace(
        go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=forecast_result.upper_ci + forecast_result.lower_ci[::-1],
            fill="toself",
            fillcolor="rgba(220,38,38,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=forecast_result.forecast,
            mode="lines",
            name=f"Forecast ({forecast_result.model_used})",
            line=dict(color="#DC2626", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Forecast — {forecast_result.model_used} "
        f"(MAPE={forecast_result.mape:.2f}%, RMSE={forecast_result.rmse:.2f})",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _fig_to_dict(fig)


def plot_model_comparison(all_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Grouped bar chart comparing RMSE, MAE, MAPE across all fitted models."""
    if not all_metrics:
        return {}

    models = list(all_metrics.keys())
    metrics = ["RMSE", "MAE", "MAPE"]
    colors = ["#2563EB", "#16A34A", "#D97706"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        values = [all_metrics[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(name=metric, x=models, y=values, marker_color=color))

    fig.update_layout(
        title="Model Comparison — Evaluation Metrics",
        xaxis_title="Model",
        yaxis_title="Error",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _fig_to_dict(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _index_to_str(series: pd.Series) -> list[str]:
    try:
        return series.index.strftime("%Y-%m-%d").tolist()
    except Exception:
        return [str(i) for i in series.index]


def _fig_to_dict(fig: go.Figure) -> dict[str, Any]:
    return json.loads(fig.to_json())
