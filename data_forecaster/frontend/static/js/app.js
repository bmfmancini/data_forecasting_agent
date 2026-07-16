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
    decisions.innerHTML = (result.decisions || []).map(function (decision) {
      var current = preflightOptions[decision.key] || decision.default || "";
      var lossLabels = {
        auto: "Auto — forecasting assistant recommends",
        rmse: "Avoid occasional large errors (RMSE)",
        mae: "Minimize the typical absolute error (MAE)",
        wape: "Control error relative to total volume (WAPE)",
        mase: "Compare accuracy against a naive forecast (MASE)"
      };
      var options = (decision.options || []).map(function (option) {
        var label = decision.key === "loss_metric" ? lossLabels[option] || option : option;
        return '<option value="' + escapeHtml(option) + '"' + (option === current ? " selected" : "") + ">" + escapeHtml(label) + "</option>";
      }).join("");
      return '<div class="card mb-3"><div class="card-body"><label class="form-label" for="pf-' + escapeHtml(decision.key) + '">' + escapeHtml(decision.label) + "</label>" +
        '<p class="small text-muted">' + escapeHtml(decision.message) + '</p><select class="form-select preflight-choice" id="pf-' + escapeHtml(decision.key) + '" data-key="' + escapeHtml(decision.key) + '">' + options + "</select></div></div>";
    }).join("");
    updatePreflightContinue();
  }

  function currentPreflightChoices() {
    var choices = {};
    document.querySelectorAll(".preflight-choice").forEach(function (select) { choices[select.dataset.key] = select.value; });
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
      "clean-smoothing": "smoothing"
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
    document.addEventListener("change", function (event) { if (event.target.classList.contains("preflight-choice")) updatePreflightContinue(); });
    var horizon = document.getElementById("inp-horizon");
    if (horizon) horizon.addEventListener("input", function () { document.getElementById("horizon-val").textContent = horizon.value; });
    ["inp-prompt", "inp-horizon", "sel-model"].forEach(function (id) { var field = document.getElementById(id); if (field) field.addEventListener(id === "inp-prompt" ? "blur" : "change", function () { saveSetupState(); }); });
    document.getElementById("btn-run").addEventListener("click", runAnalysis);
    if (window.forecastUploadInfo) { populateColumnSelectors(window.forecastUploadInfo); setUploadStatus(window.forecastUploadInfo.rows + " rows ready.", false); }
  }

  document.addEventListener("DOMContentLoaded", init);
  window.App = { triggerPreflight: triggerPreflight, populateColumnSelectors: populateColumnSelectors };
}());
