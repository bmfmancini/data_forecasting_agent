/**
 * Plotly chart rendering helpers.
 *
 * Finds every element with the ``chart-container`` class and renders the
 * Plotly figure described by its ``data-chart`` attribute.  For chat
 * visualisation elements with ``data-chart-type="pie"`` a simple pie chart
 * is built from ``data-chart`` containing ``{labels, values}``.
 */

(function () {
  "use strict";

  var DARK_LAYOUT_OVERRIDES = {
    paper_bgcolor: "#0e1117",
    plot_bgcolor: "#0e1117",
    font: { color: "#e6edf3", size: 12 },
    xaxis: {
      gridcolor: "#30363d",
      linecolor: "#30363d",
      zerolinecolor: "#30363d",
    },
    yaxis: {
      gridcolor: "#30363d",
      linecolor: "#30363d",
      zerolinecolor: "#30363d",
    },
    legend: { bgcolor: "rgba(0,0,0,0)" },
    margin: { l: 50, r: 20, t: 40, b: 50 },
  };

  /**
   * Merge two plain objects (shallow, second overwrites first).
   *
   * @param {object} base
   * @param {object} overrides
   * @returns {object}
   */
  function mergeLayout(base, overrides) {
    var result = Object.assign({}, base || {});
    Object.keys(overrides).forEach(function (key) {
      if (
        typeof overrides[key] === "object" &&
        overrides[key] !== null &&
        !Array.isArray(overrides[key])
      ) {
        result[key] = Object.assign({}, result[key] || {}, overrides[key]);
      } else {
        result[key] = overrides[key];
      }
    });
    return result;
  }

  /**
   * Render a Plotly figure inside a container element.
   *
   * @param {HTMLElement} el   - Container element.
   * @param {object}      fig  - Plotly figure dict ``{data, layout}``.
   */
  function renderPlotly(el, fig) {
    var layout = mergeLayout(fig.layout || {}, DARK_LAYOUT_OVERRIDES);
    Plotly.newPlot(el, fig.data || [], layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
    });
  }

  /**
   * Render a pie chart from ``{labels, values}`` data.
   *
   * @param {HTMLElement} el   - Container element.
   * @param {object}      data - Object with ``labels`` and ``values`` arrays.
   */
  function renderPie(el, data) {
    var layout = mergeLayout({}, DARK_LAYOUT_OVERRIDES);
    Plotly.newPlot(
      el,
      [{ type: "pie", labels: data.labels || [], values: data.values || [] }],
      layout,
      { responsive: true, displayModeBar: false, displaylogo: false }
    );
  }

  /**
   * Attempt to render a chart from a dynamic LLM-generated config.
   *
   * Supports the same ``chart_type`` values as the Streamlit
   * ``DynamicVisualizer``: line, bar, scatter, histogram, box, violin,
   * heatmap, area.
   *
   * @param {HTMLElement} el   - Container element.
   * @param {object}      cfg  - Dynamic visualisation config from the backend.
   */
  function renderDynamic(el, cfg) {
    var chartType = (cfg.chart_type || "bar").toLowerCase();
    var xData = cfg.x_data || [];
    var yData = cfg.y_data || [];
    var trace = { x: xData, y: yData };

    if (chartType === "line" || chartType === "area") {
      trace.type = "scatter";
      trace.mode = "lines" + (chartType === "area" ? "+markers" : "");
      if (chartType === "area") trace.fill = "tozeroy";
    } else if (chartType === "scatter") {
      trace.type = "scatter";
      trace.mode = "markers";
    } else if (chartType === "histogram") {
      trace = { x: xData, type: "histogram" };
    } else if (chartType === "box") {
      trace = { y: yData.length ? yData : xData, type: "box" };
    } else if (chartType === "violin") {
      trace = { y: yData.length ? yData : xData, type: "violin" };
    } else if (chartType === "heatmap") {
      trace = { z: cfg.z_data || [], type: "heatmap" };
    } else {
      trace.type = "bar";
    }

    if (cfg.name) trace.name = cfg.name;
    if (cfg.color) trace.marker = { color: cfg.color };

    var layout = mergeLayout(
      {
        title: cfg.title || "",
        xaxis: { title: cfg.x_label || "" },
        yaxis: { title: cfg.y_label || "" },
      },
      DARK_LAYOUT_OVERRIDES
    );

    Plotly.newPlot(el, [trace], layout, {
      responsive: true,
      displayModeBar: false,
      displaylogo: false,
    });
  }

  /** Render all ``.chart-container`` elements on the current page. */
  function renderAll() {
    var containers = document.querySelectorAll(".chart-container");
    containers.forEach(function (el) {
      var rawChart = el.dataset.chart;
      var chartType = el.dataset.chartType || "plotly";
      if (!rawChart) return;

      // Show loading shimmer while parsing/plotting
      el.classList.add("is-loading");

      var parsed;
      try {
        parsed = JSON.parse(rawChart);
      } catch (e) {
        el.classList.remove("is-loading");
        return;
      }

      // Use rAF so the shimmer paints before Plotly blocks
      requestAnimationFrame(function () {
        if (chartType === "pie") {
          renderPie(el, parsed);
        } else if (chartType === "dynamic") {
          renderDynamic(el, parsed);
        } else {
          renderPlotly(el, parsed);
        }
        el.classList.remove("is-loading");
      });
    });
  }

  document.addEventListener("DOMContentLoaded", renderAll);

  window.Charts = { renderAll: renderAll, renderPlotly: renderPlotly, renderPie: renderPie };
}());
