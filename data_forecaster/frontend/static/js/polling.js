/**
 * Job status polling module.
 *
 * Polls the Flask ``/api/jobs/status`` endpoint every 1.5 seconds while
 * a forecast job is running.  On completion the browser is redirected to
 * the report tab; on error the sidebar shows the error message.
 */

(function () {
  "use strict";

  var _intervalId = null;
  var POLL_INTERVAL_MS = 1500;

  function updateBar(elementId, progress) {
    var bar = document.getElementById(elementId);
    if (bar) {
      bar.style.width = progress + "%";
      bar.classList.add("active-polling");
      if (elementId === "progress-bar-main") bar.textContent = progress + "%";
    }
  }


  /**
   * Update the progress bar and step text in the sidebar.
   *
   * @param {number} progress  Completion percentage (0–100).
   * @param {string} stepText  Human-readable step description.
   */
  function updateProgress(progress, stepText) {
    var text = document.getElementById("progress-text");
    if (text) {
        text.innerHTML = ''; // Clear existing content
        var dot = document.createElement('span');
        dot.className = 'job-status-dot';
        text.appendChild(dot);
        var textNode = document.createTextNode(" " + stepText + " (" + progress + "%)");
        text.appendChild(textNode);
    }

    updateBar("progress-bar", progress);
    updateBar("progress-bar-main", progress);
    // Also update the main progress bar on the setup page
    var mainStep = document.getElementById("progress-step");
    if (mainStep) mainStep.textContent = stepText;
  }

  /**
   * Show a job-level error message in the sidebar and reset the run button.
   *
   * @param {string} message  Error description safe for display.
   */
  function showError(message) {
    var area = document.getElementById("progress-area");
    if (area) area.classList.add("d-none");

    var runBtn = document.getElementById("btn-run");
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "Run Analysis";
    }

    var errEl = document.querySelector(".polling-error-msg");
    if (!errEl) {
      errEl = document.createElement("div");
      errEl.className =
        "alert alert-danger mt-2 py-1 px-2 small polling-error-msg";
      var sidebar = document.querySelector(".sidebar-inner");
      if (sidebar) sidebar.appendChild(errEl);
    }
    errEl.textContent = message;
  }

  /** Execute a single poll cycle. */
  function poll() {
    fetch("/api/jobs/status", {
      headers: { "X-CSRFToken": "" },
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          stop();
          showError(data.error);
          return;
        }

        updateProgress(data.progress || 0, data.step || "Processing...");

        if (data.done) {
          stop();
          if (data.redirect) {
            window.location.href = data.redirect;
          } else {
            window.location.reload();
          }
        } else if (data.status === "error") {
          stop();
          showError(data.error || "Analysis failed.");
        }
      })
      .catch(function (err) {
        stop();
        showError("Status poll failed: " + err);
      });
  }

  /** Start the polling interval. */
  function start() {
    if (_intervalId !== null) return;
    _intervalId = setInterval(poll, POLL_INTERVAL_MS);
    var progressArea = document.getElementById("progress-area");
    if (progressArea) progressArea.classList.remove("d-none");
  }

  /** Stop the polling interval. */
  function stop() {
    if (_intervalId !== null) {
      clearInterval(_intervalId);
      _intervalId = null;
    }
    // Remove animated stripes when done
    var bars = document.querySelectorAll(".progress-bar.active-polling");
    bars.forEach(function (b) { b.classList.remove("active-polling"); });
  }

  /** Auto-start polling when the page loads if a job is already running. */
  document.addEventListener("DOMContentLoaded", function () {
    var body = document.body;
    var isRunning = body && body.dataset.jobRunning === "true";
    if (isRunning) {
      var progress = parseInt(body.dataset.jobProgress || "0", 10);
      var step = body.dataset.jobStep || "Processing...";
      updateProgress(progress, step);
      start();
    }
  });

  window.Polling = { start: start, stop: stop };
}());
