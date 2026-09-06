/** Forecast setup wizard, upload, preflight, and analysis submission. */
(function () {
  "use strict";

  var wizardStep = 1;
  var preflight = null;
  var preflightOptions = {};

  function csrfToken() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
      body: JSON.stringify(body)
    });
  }

  function escapeHtml(value) {
    var node = document.createElement("div");
    node.textContent = String(value || "");
    return node.innerHTML;
  }

  function showStep(step) {
    wizardStep = step;
    document.querySelectorAll("[data-wizard-step]").forEach(function (panel) {
      panel.classList.toggle("d-none", Number(panel.dataset.wizardStep) !== step);
    });
    document.querySelectorAll("#setup-stepper .stepper-step").forEach(function (item) {
      var number = Number(item.dataset.step);
      item.classList.toggle("active", number === step);
      item.classList.toggle("completed", number < step);
      if (number === step) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function setUploadStatus(message, isError) {
    var element = document.getElementById("upload-status");
    if (!element) return;
    element.textContent = message;
    element.className = "mt-2 small " + (isError ? "text-danger" : "text-success");
  }

  function updateContinueState() {
    var date = document.getElementById("sel-date");
    var value = document.getElementById("sel-value");
    var button = document.getElementById("btn-to-preflight");
    if (button) button.disabled = !date || !value || date.disabled || !date.value || !value.value;
  }

  function populateColumnSelectors(info) {
    var date = document.getElementById("sel-date");
    var value = document.getElementById("sel-value");
    if (!date || !value) return;
    [date, value].forEach(function (select) {
      select.innerHTML = "";
      select.disabled = false;
    });
    (info.columns || []).forEach(function (column) {
      date.options.add(new Option(column, column, false, column === info.detected_date_col));
      value.options.add(new Option(column, column, false, column === info.detected_value_col));
    });
    var frequency = document.getElementById("detected-frequency");
    if (frequency) frequency.textContent = info.detected_frequency || "—";
    updateContinueState();
  }

  function uploadFile(file) {
    setUploadStatus("Uploading…", false);
    var form = new FormData();
    form.append("file", file);
    fetch("/api/upload", { method: "POST", headers: { "X-CSRFToken": csrfToken() }, body: form })
      .then(function (response) { return response.json(); })
      .then(function (data) {
        if (data.error) { setUploadStatus(data.error, true); return; }
        preflight = null;
        preflightOptions = {};
        setUploadStatus("Uploaded — " + data.rows + " rows detected.", false);
        populateColumnSelectors(data);
      })
      .catch(function (error) { setUploadStatus("Upload failed: " + error, true); });
  }

  function renderPreflight(result) {
    var status = document.getElementById("preflight-status");
    var decisions = document.getElementById("preflight-decisions");
    if (!status || !decisions) return;
    var messages = (result.issues || []).concat(result.warnings || [], result.errors || []);
    var tone = result.status === "error" ? "danger" : result.status === "warning" ? "warning" : "success";
    var title = result.status === "error" ? "Preflight issues found" : result.status === "warning" ? "Preflight cautions" : "Preflight ready";
    status.innerHTML = '<div class="alert alert-' + tone + '"><strong>' + title + "</strong>" +
      (result.detected_frequency ? '<p class="mb-0 mt-2 small">Detected frequency: <strong>' + escapeHtml(result.detected_frequency) + "</strong></p>" : "") +
      (messages.length ? "<ul class=\"mb-0 mt-2\">" + messages.map(function (message) { return "<li>" + escapeHtml(message) + "</li>"; }).join("") + "</ul>" : "") + "</div>";
    decisions.innerHTML = (result.decisions || []).map(renderDecision).join("");
    updateHolidaySubdivision(preflightOptions.holidays_subdivision || "");
    updatePreflightContinue();
  }

  var _LOSS_LABELS = {
    auto: "Auto — forecasting assistant recommends",
    rmse: "Avoid occasional large errors (RMSE)",
    mae: "Minimize the typical absolute error (MAE)",
    wape: "Control error relative to total volume (WAPE)",
    mase: "Compare accuracy against a naive forecast (MASE)",
    pinball: "Choose a quantile for unequal error costs (pinball loss)"
  };

  var _DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

  function decisionLabel(decision, option) {
    if (decision.key === "loss_metric") return _LOSS_LABELS[option] || option;
    if (decision.kind === "country") {
      var idx = (decision.options || []).indexOf(option);
      return (decision.option_labels && idx >= 0) ? decision.option_labels[idx] : option;
    }
    return option;
  }

  // Custom structured inputs are parsed into JSON at submit time (below); the
  // plain <select> for kind=="select"/"country" is read directly by value.
  function renderDecision(decision) {
    var current = preflightOptions[decision.key] !== undefined ? preflightOptions[decision.key] : decision.default;
    var id = "pf-" + decision.key;
    var head = '<div class="card mb-3"><div class="card-body"><label class="form-label" for="' + escapeHtml(id) + '">' + escapeHtml(decision.label) + "</label>" +
      '<p class="small text-muted">' + escapeHtml(decision.message) + "</p>";
    var tail = "</div></div>";
    if (decision.detail_key) {
      var detailId = id + "-details";
      tail = '<label class="form-label mt-3" for="' + escapeHtml(detailId) + '">Details (optional)</label>' +
        '<textarea class="form-control preflight-text" id="' + escapeHtml(detailId) + '" data-key="' + escapeHtml(decision.detail_key) + '" rows="3" placeholder="' + escapeHtml(decision.detail_placeholder || "") + '">' + escapeHtml(preflightOptions[decision.detail_key] || "") + '</textarea>' + tail;
    }
    if (decision.kind === "country") {
      tail = '<div id="pf-subdivision-group" class="mt-3" hidden><label class="form-label" for="pf-holidays_subdivision">State / province / region</label>' +
        '<select class="form-select preflight-choice" id="pf-holidays_subdivision" data-key="holidays_subdivision"></select>' +
        '<p class="small text-muted mt-2">Select the region whose holidays affect this series, or use the country calendar without a regional selection.</p></div>' + tail;
    }
    if (decision.kind === "dates") {
      return head + '<textarea class="form-control preflight-dates" id="' + escapeHtml(id) + '" data-key="' + escapeHtml(decision.key) + '" rows="4" placeholder="2024-11-29, spike, Black Friday&#10;2025-01-01, holiday, New Year">' + escapeHtml(formatEvents(current)) + "</textarea>" + tail;
    }
    if (decision.kind === "covariates") {
      return head + '<textarea class="form-control preflight-covariates" id="' + escapeHtml(id) + '" data-key="' + escapeHtml(decision.key) + '" rows="4" placeholder="price: 2024-01-01=9.99, 2024-02-01=9.99&#10;promo: 2024-11-29=1, 2024-12-01=0">' + escapeHtml(formatCovariates(current)) + "</textarea>" + tail;
    }
    // kind == "select" or "country": a plain dropdown (country codes carry labels).
    var options = (decision.options || []).map(function (option) {
      return '<option value="' + escapeHtml(option) + '"' + (option === current ? " selected" : "") + ">" + escapeHtml(decisionLabel(decision, option)) + "</option>";
    }).join("");
    var placeholder = decision.kind === "country" ? '<option value=""' + (current === "" || current === undefined ? " selected" : "") + ">No holiday calendar</option>" : "";
    return head + '<select class="form-select preflight-choice" id="' + escapeHtml(id) + '" data-key="' + escapeHtml(decision.key) + '">' + placeholder + options + "</select>" + tail;
  }

  function updateHolidaySubdivision(selected) {
    var country = document.getElementById("pf-holidays_country");
    var region = document.getElementById("pf-holidays_subdivision");
    var group = document.getElementById("pf-subdivision-group");
    if (!country || !region || !group) return;
    var decision = (preflight.decisions || []).find(function (item) { return item.key === "holidays_country"; });
    var regions = ((decision && decision.subdivisions) || {})[country.value] || [];
    region.innerHTML = '<option value="">Country calendar only</option>' + regions.map(function (item) {
      return '<option value="' + escapeHtml(item.code) + '"' + (item.code === selected ? ' selected' : '') + '>' + escapeHtml(item.label) + '</option>';
    }).join("");
    group.hidden = !country.value || !regions.length;
    region.disabled = group.hidden;
  }

  function formatEvents(value) {
    if (!Array.isArray(value) || !value.length) return "";
    return value.map(function (event) {
      if (!event || !event.date) return "";
      var line = event.date + ", " + (event.type || "intervention");
      if (event.label && event.label !== event.type) line += ", " + event.label;
      return line;
    }).filter(Boolean).join("\n");
  }

  function parseEvents(text) {
    var events = [];
    (text || "").split(/\n+/).forEach(function (raw) {
      var line = raw.trim();
      if (!line) return;
      var parts = line.split(",").map(function (p) { return p.trim(); });
      if (!_DATE_RE.test(parts[0] || "")) return;
      var type = (parts[1] || "intervention").toLowerCase() || "intervention";
      var label = parts.slice(2).join(",").trim() || type;
      events.push({ type: type, date: parts[0], label: label });
    });
    return events;
  }

  function formatCovariates(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return "";
    return Object.keys(value).map(function (name) {
      var series = value[name] || {};
      var pairs = Object.keys(series).map(function (date) { return date + "=" + series[date]; }).join(", ");
      return name + ": " + pairs;
    }).join("\n");
  }

  function parseCovariates(text) {
    var covariates = {};
    (text || "").split(/\n+/).forEach(function (raw) {
      var line = raw.trim();
      if (!line || line.indexOf(":") === -1) return;
      var split = line.split(/:(.*)/);
      var name = split[0].trim();
      var rest = split[1] || "";
      if (!name) return;
      var series = {};
      rest.split(",").forEach(function (pair) {
        var kv = pair.split("=");
        var date = (kv[0] || "").trim();
        var val = parseFloat((kv[1] || "").trim());
        if (_DATE_RE.test(date) && isFinite(val)) series[date] = val;
      });
      if (Object.keys(series).length) covariates[name] = series;
    });
    return covariates;
  }

  function currentPreflightChoices() {
    var choices = {};
    document.querySelectorAll(".preflight-choice").forEach(function (select) { choices[select.dataset.key] = select.value; });
    document.querySelectorAll(".preflight-text").forEach(function (area) { choices[area.dataset.key] = area.value.trim(); });
    document.querySelectorAll(".preflight-dates").forEach(function (area) { choices[area.dataset.key] = parseEvents(area.value); });
    document.querySelectorAll(".preflight-covariates").forEach(function (area) { choices[area.dataset.key] = parseCovariates(area.value); });
    return choices;
  }

  function updatePreflightContinue() {
    var button = document.getElementById("btn-to-configure");
    if (!button) return;
    var choices = currentPreflightChoices();
    var blocked = Object.keys(choices).some(function (key) { return choices[key] === "stop"; });
    button.disabled = !preflight || preflight.status === "error" || blocked;
  }

  function triggerPreflight() {
    var date = document.getElementById("sel-date");
    var value = document.getElementById("sel-value");
    if (!date || !value || !date.value || !value.value) return Promise.reject(new Error("Choose both columns first."));
    return postJSON("/api/columns", { date_col: date.value, value_col: value.value })
      .then(function (response) { return response.json(); })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        preflight = data.preflight;
        preflightOptions = {};
        (preflight.decisions || []).forEach(function (decision) { preflightOptions[decision.key] = decision.default; });
        renderPreflight(preflight);
        return preflight;
      });
  }

  function saveSetupState() {
    var horizon = document.getElementById("inp-horizon");
    var model = document.getElementById("sel-model");
    var prompt = document.getElementById("inp-prompt");
    return postJSON("/api/setup-state", {
      forecast_horizon: horizon ? horizon.value : 12,
      model_choice: model ? model.value : "Auto (AI selects)",
      user_prompt: prompt ? prompt.value : ""
    });
  }

  function collectStatisticalTuning() {
    var disabled = [];
    document.querySelectorAll(".stat-tuning-toggle").forEach(function (item) {
      if (!item.checked && item.dataset.statTest) disabled.push(item.dataset.statTest);
    });
    return { disabled_tests: disabled };
  }

  /** Return only explicit cleaning overrides, preserving preflight choices otherwise. */
  function collectCleaningOptions() {
    var fields = {
      "clean-frequency": "frequency",
      "clean-duplicates": "duplicate_strategy",
      "clean-missing": "missing_strategy",
      "clean-outliers": "outlier_strategy",
      "clean-smoothing": "smoothing",
      "forecast-aggregation": "aggregation",
      "forecast-demand-pattern": "demand_pattern",
      "forecast-quantile": "forecast_quantile"
    };
    var options = {};
    Object.keys(fields).forEach(function (id) {
      var field = document.getElementById(id);
      if (!field) return;
      // "Let AI Decide" deliberately defers to the preflight selection.
      if (field.value !== "Let AI Decide") options[fields[id]] = field.value;
    });
    return options;
  }

  function showRunError(message) {
    var element = document.getElementById("run-error-message");
    var button = document.getElementById("btn-run");
    if (button) { button.disabled = false; button.textContent = "Run forecast"; }
    if (element) { element.textContent = message; element.className = "alert alert-danger mt-3"; element.style.display = "block"; }
  }

  function runAnalysis() {
    var date = document.getElementById("sel-date");
    var value = document.getElementById("sel-value");
    var horizon = document.getElementById("inp-horizon");
    var model = document.getElementById("sel-model");
    var prompt = document.getElementById("inp-prompt");
    var button = document.getElementById("btn-run");
    if (!date || !value) return;
    if (button) { button.disabled = true; button.textContent = "Starting forecast…"; }
    var options = Object.assign(
      {}, preflightOptions, currentPreflightChoices(), collectCleaningOptions(),
      { statistical_tuning: collectStatisticalTuning() }
    );
    postJSON("/api/analyze", { date_col: date.value, value_col: value.value, forecast_horizon: Number(horizon.value), model_choice: model.value, user_prompt: prompt.value, preflight_options: options })
      .then(function (response) {
        if (response.status === 202) { window.location.assign("/forecast-progress"); return; }
        return response.json().then(function (data) { throw new Error(data.error || "Failed to submit forecast."); });
      })
      .catch(function (error) { showRunError(error.message || String(error)); });
  }

  function init() {
    var input = document.getElementById("file-input");
    if (input) input.addEventListener("change", function () { if (input.files && input.files[0]) uploadFile(input.files[0]); });
    ["sel-date", "sel-value"].forEach(function (id) { var select = document.getElementById(id); if (select) select.addEventListener("change", updateContinueState); });
    document.getElementById("btn-to-preflight").addEventListener("click", function () {
      this.disabled = true; this.textContent = "Running checks…";
      triggerPreflight().then(function () { showStep(2); }).catch(function (error) { alert(error.message || String(error)); }).finally(function () { var button = document.getElementById("btn-to-preflight"); button.textContent = "Continue to preflight"; updateContinueState(); });
    });
    document.getElementById("btn-to-configure").addEventListener("click", function () {
      preflightOptions = currentPreflightChoices();
      postJSON("/api/preflight-choices", { choices: preflightOptions }).then(function () { showStep(3); });
    });
    document.querySelectorAll("[data-wizard-back]").forEach(function (button) { button.addEventListener("click", function () { showStep(Number(button.dataset.wizardBack)); }); });
    document.addEventListener("change", function (event) { if (event.target.id === "pf-holidays_country") updateHolidaySubdivision(""); if (event.target.classList.contains("preflight-choice")) updatePreflightContinue(); });
    var horizon = document.getElementById("inp-horizon");
    if (horizon) horizon.addEventListener("input", function () { document.getElementById("horizon-val").textContent = horizon.value; });
    ["inp-prompt", "inp-horizon", "sel-model"].forEach(function (id) { var field = document.getElementById(id); if (field) field.addEventListener(id === "inp-prompt" ? "blur" : "change", function () { saveSetupState(); }); });
    document.getElementById("btn-run").addEventListener("click", runAnalysis);
    if (window.forecastUploadInfo) { populateColumnSelectors(window.forecastUploadInfo); setUploadStatus(window.forecastUploadInfo.rows + " rows ready.", false); }
  }

  document.addEventListener("DOMContentLoaded", init);
  window.App = { triggerPreflight: triggerPreflight, populateColumnSelectors: populateColumnSelectors };
}());
