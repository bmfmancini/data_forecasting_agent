/**
 * Main application JavaScript.
 *
 * Handles file upload via AJAX, column selection changes, preflight
 * option persistence, and the Run Analysis submission.  All AJAX
 * requests include the CSRF token read from the page meta tag.
 */

(function () {
  "use strict";

  /** Read the CSRF token from the page meta tag. */
  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  /**
   * POST JSON to a URL with the CSRF token header.
   *
   * @param {string} url
   * @param {object} body
   * @returns {Promise<Response>}
   */
  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCsrfToken(),
      },
      body: JSON.stringify(body),
    });
  }

  /**
   * POST a FormData object (for file uploads) with the CSRF token header.
   *
   * @param {string} url
   * @param {FormData} formData
   * @returns {Promise<Response>}
   */
  function postForm(url, formData) {
    return fetch(url, {
      method: "POST",
      headers: { "X-CSRFToken": getCsrfToken() },
      body: formData,
    });
  }

  /** Display a short status message below the upload zone. */
  function setUploadStatus(message, isError) {
    const el = document.getElementById("upload-status");
    if (!el) return;
    el.textContent = message;
    el.className = "mt-2 small " + (isError ? "text-danger" : "text-success");
  }

  /**
   * Update the column dropdowns after a successful upload.
   *
   * @param {object} info - Upload response from the backend.
   */
  function populateColumnSelectors(info) {
    const dateSel = document.getElementById("sel-date");
    const valueSel = document.getElementById("sel-value");
    if (!dateSel || !valueSel) return;

    [dateSel, valueSel].forEach(function (sel) {
      sel.innerHTML = "";
      sel.disabled = false;
    });

    (info.columns || []).forEach(function (col) {
      dateSel.options.add(new Option(col, col, false, col === info.detected_date_col));
      valueSel.options.add(new Option(col, col, false, col === info.detected_value_col));
    });

    const freqEl = document.querySelector("p.small.text-muted strong");
    if (freqEl) freqEl.textContent = info.detected_frequency || "—";

    enableControls();
  }

  /** Enable all forecast configuration controls that depend on an uploaded file. */
  function enableControls() {
    ["sel-model", "inp-horizon", "inp-prompt", "btn-run"].forEach(function (id) {
      const el = document.getElementById(id);
      if (el) el.disabled = false;
    });
  }

  /**
   * Handle a new file selection from the file input.
   *
   * @param {File} file
   */
  function uploadFile(file) {
    setUploadStatus("Uploading...", false);
    const formData = new FormData();
    formData.append("file", file);

    postForm("/api/upload", formData)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          setUploadStatus(data.error, true);
        } else {
          setUploadStatus(
            "Uploaded — " + data.rows + " rows detected.",
            false
          );
          populateColumnSelectors(data);
          triggerPreflight();
        }
      })
      .catch(function (err) {
        setUploadStatus("Upload failed: " + err, true);
      });
  }

  /** Call the preflight endpoint with the current column selection. */
  function triggerPreflight() {
    const dateSel = document.getElementById("sel-date");
    const valueSel = document.getElementById("sel-value");
    if (!dateSel || !valueSel || dateSel.disabled) return;

    // Show a loading state for preflight
    const statusEl = document.getElementById("preflight-status");
    if(statusEl) {
        statusEl.innerHTML = `<div class="alert alert-secondary">Running preflight checks...</div>`;
    }

    postJSON("/api/columns", {
      date_col: dateSel.value,
      value_col: valueSel.value,
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.preflight) {
          updatePreflightStatus(data.preflight);
          checkPreflightBlocks(data.preflight);
        } else if (data.error) {
            updatePreflightStatus({ status: 'error', errors: [data.error]});
        }
      })
      .catch(function (err) {
        updatePreflightStatus({ status: 'error', errors: [`Connection error: ${err}`]});
      });
  }

  /**
   * Update the preflight status display on the setup page.
   *
   * @param {object} preflight
   */
  function updatePreflightStatus(preflight) {
    const statusEl = document.getElementById("preflight-status");
    if (!statusEl) return;

    let alertClass = "alert-success";
    let title = "Preflight Ready";
    let messages = preflight.issues || [];

    if (preflight.status === "warning") {
        alertClass = "alert-warning";
        title = "Preflight Cautions";
        messages = messages.concat(preflight.warnings);
    } else if (preflight.status === "error") {
        alertClass = "alert-danger";
        title = "Preflight Issues Found";
        messages = messages.concat(preflight.errors);
    }

    let html = `<div class="alert ${alertClass}"><strong>${title}</strong>`;
    if (messages.length > 0) {
        html += '<ul>';
        messages.forEach(msg => { html += `<li>${msg}</li>`; });
        html += '</ul>';
    }

    if (preflight.decisions && preflight.decisions.length > 0) {
      html +=
        '<button type="button" class="btn btn-outline-info btn-sm w-100 mt-2" ' +
        'data-bs-toggle="modal" data-bs-target="#preflightModal">' +
        "Review Preflight Options</button>";
      populatePreflightModal(preflight);
    }
    html += '</div>';
    statusEl.innerHTML = html;
  }

  /**
   * Populate the preflight modal body from live preflight data returned by the
   * server.
   *
   * @param {object} preflight  Preflight result object from /api/columns.
   */
  function populatePreflightModal(preflight) {
    const body = document.getElementById("preflight-modal-body");
    if (!body) return;

    let html = "";

    if (preflight.detected_frequency) {
      html += '<p class="text-muted small">Detected frequency: <strong>' +
        preflight.detected_frequency + "</strong></p>";
    }

    (preflight.issues || []).forEach(function (msg) {
      html += '<div class="alert alert-info py-1 px-2 small">' + msg + "</div>";
    });
    (preflight.warnings || []).forEach(function (msg) {
      html += '<div class="alert alert-warning py-1 px-2 small">' + msg + "</div>";
    });

    const saved = App._preflightOptions || {};
    (preflight.decisions || []).forEach(function (d) {
      const current = saved[d.key] || d.default || "";
      let opts = "";
      (d.options || []).forEach(function (opt) {
        const sel = opt === current ? " selected" : "";
        opts += '<option value="' + opt + '"' + sel + '>' +
          opt.charAt(0).toUpperCase() + opt.slice(1) + "</option>";
      });
      html +=
        '<div class="mb-3">' +
        '<label class="form-label" for="pf-' + d.key + '">' + d.label + "</label>" +
        '<p class="text-muted small">' + d.message + "</p>" +
        '<select class="form-select form-select-sm preflight-choice" id="pf-' +
        d.key + '" data-key="' + d.key + '">' +
        opts +
        "</select></div>";
    });

    body.innerHTML = html;
  }

  /**
   * Disable the Run Analysis button when a preflight decision blocks the run.
   *
   * @param {object} preflight
   */
  function checkPreflightBlocks(preflight) {
    const runBtn = document.getElementById("btn-run");
    if (!runBtn) return;
    const opts = App._preflightOptions || {};
    const blocks =
      preflight.decisions &&
      preflight.decisions.some(function (d) {
        return (opts[d.key] || d.default) === "stop";
      });
    runBtn.disabled = blocks;
  }

  /** Submit the analysis job via AJAX and start the polling loop. */
  function runAnalysis() {
    const dateSel = document.getElementById("sel-date");
    const valueSel = document.getElementById("sel-value");
    const horizonEl = document.getElementById("inp-horizon");
    const modelEl = document.getElementById("sel-model");
    const promptEl = document.getElementById("inp-prompt");
    const runBtn = document.getElementById("btn-run");

    if (!dateSel || !valueSel) return;

    const payload = {
      date_col: dateSel.value,
      value_col: valueSel.value,
      forecast_horizon: horizonEl ? parseInt(horizonEl.value, 10) : 12,
      model_choice: modelEl ? modelEl.value : "Auto (AI selects)",
      user_prompt: promptEl ? promptEl.value : "",
      preflight_options: App._preflightOptions || {},
    };

    if (runBtn) {
      runBtn.disabled = true;
      runBtn.textContent = "Running...";
    }

    const progressContainer = document.getElementById("progress-area-container");
    if (progressContainer) progressContainer.style.display = "block";
    const sidebarProgress = document.getElementById("progress-area");
    if(sidebarProgress) sidebarProgress.classList.remove("d-none");


    postJSON("/api/analyze", payload)
      .then(function (r) {
        if (r.status === 202) {
          Polling.start();
        } else {
          return r.json().then(function (data) {
            showRunError(data.error || "Failed to submit job.");
          });
        }
      })
      .catch(function (err) {
        showRunError("Connection error: " + err);
      });
  }

  /**
   * Display an inline error from the analysis submission.
   *
   * @param {string} message
   */
  function showRunError(message) {
    const runBtn = document.getElementById("btn-run");
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "Run Analysis";
    }

    const progressContainer = document.getElementById("progress-area-container");
    if (progressContainer) progressContainer.style.display = "none";
    const sidebarProgress = document.getElementById("progress-area");
    if(sidebarProgress) sidebarProgress.classList.add("d-none");


    let errEl = document.getElementById("run-error-message");
    if (!errEl) {
      errEl = document.createElement("div");
      errEl.id = "run-error-message";
      errEl.className = "alert alert-danger mt-3";
      const container = document.querySelector("#progress-area-container");
      if(container) container.insertAdjacentElement('beforebegin', errEl);
    }
    errEl.textContent = message;
    errEl.style.display = "block";
  }

  /**
   * Collect preflight decision choices from the modal selects and POST them.
   */
  function savePreflightChoices() {
    const selects = document.querySelectorAll(".preflight-choice");
    const choices = {};
    selects.forEach(function (sel) {
      choices[sel.dataset.key] = sel.value;
    });
    App._preflightOptions = choices;

    postJSON("/api/preflight-choices", { choices: choices })
      .then(function (r) { return r.json(); })
      .then(function () {
        const modal = bootstrap.Modal.getInstance(
          document.getElementById("preflightModal")
        );
        if (modal) modal.hide();

        const blocks = Object.values(choices).some(function (v) {
          return v === "stop";
        });
        const runBtn = document.getElementById("btn-run");
        if (runBtn) runBtn.disabled = blocks;
      })
      .catch(function () {});
  }

  /** Wire up the forecast horizon range slider display. */
  function initHorizonSlider() {
    const slider = document.getElementById("inp-horizon");
    const label = document.getElementById("horizon-val");
    if (!slider || !label) return;
    slider.addEventListener("input", function () {
      label.textContent = slider.value;
    });
  }

  /** Wire up the file input change event. */
  function initFileInput() {
    const input = document.getElementById("file-input");
    if (!input) return;
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) {
        uploadFile(input.files[0]);
      }
    });
  }

  /** Wire up column selector change events to trigger preflight. */
  function initColumnSelectors() {
    ["sel-date", "sel-value"].forEach(function (id) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", function () {
          triggerPreflight();
        });
      }
    });
  }

  /** Wire up the Run Analysis button. */
  function initRunButton() {
    const runBtn = document.getElementById("btn-run");
    if (runBtn) {
      runBtn.addEventListener("click", function () {
        runAnalysis();
      });
    }
  }

  function init() {
    // Only run initializers if the relevant elements are on the page
    if (document.getElementById('inp-horizon')) initHorizonSlider();
    if (document.getElementById('file-input')) initFileInput();
    if (document.getElementById('sel-date')) initColumnSelectors();
    if (document.getElementById('btn-run')) initRunButton();
  }

  document.addEventListener("DOMContentLoaded", init);

  window.App = {
    runAnalysis: runAnalysis,
    savePreflightChoices: savePreflightChoices,
    triggerPreflight: triggerPreflight,
    populateColumnSelectors: populateColumnSelectors,
    enableControls: enableControls,
    updateUploadStatus: setUploadStatus, // Renamed for clarity in template
    _preflightOptions: {},
  };
}());
