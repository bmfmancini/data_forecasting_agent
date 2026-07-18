/**
 * Job status polling module.
 *
 * Polls the Flask ``/api/jobs/status`` endpoint every 1.5 seconds while
 * a forecast job is running.  On completion the browser is redirected to
 * the report tab; on error the sidebar shows the error message.
 */

(function () {
  "use strict";

  var _pollTimer = null;
  var _pollInFlight = false;
  var _terminal = false;
  var _started = false;
  var POLL_INTERVAL_MS = 1500;
  var MAX_BACKOFF_MS = 10000;
  var _currentInterval = POLL_INTERVAL_MS;

  function updateBar(elementId, progress) {
    var bar = document.getElementById(elementId);
    if (bar) {
      bar.style.width = progress + "%";
      bar.setAttribute("aria-valuenow", progress);
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

  function formatDuration(value) {
    if (value === null || typeof value === "undefined") return "—";
    var seconds = Math.max(0, parseInt(value, 10) || 0);
    if (seconds < 60) return seconds + "s";
    if (seconds < 3600) return Math.floor(seconds / 60) + "m " + (seconds % 60) + "s";
    return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
  }

  function updateHeartbeat(data) {
    var state = data.liveness || (data.status === "pending" ? "queued" : "active");
    var titles = {
      queued: "Queued for a forecast worker",
      active: "Forecast worker is active",
      delayed: "Worker signal is delayed",
      stale: "Worker status needs attention",
      terminal: "Forecast processing finished"
    };
    var details = {
      queued: "The job is safely queued and will start when capacity is available.",
      active: "The backend is still processing this forecast, even if the percentage has not changed.",
      delayed: "The last worker signal is later than expected. The job has not been marked failed.",
      stale: "The worker signal is stale. The system will keep checking; this warning is not a failure result.",
      terminal: "The backend reported a terminal job state."
    };
    var panel = document.getElementById("job-heartbeat");
    if (panel) {
      panel.className = "job-heartbeat job-heartbeat-" + state + " mt-4";
    }
    var title = document.getElementById("heartbeat-title");
    if (title) title.textContent = titles[state] || titles.active;
    var detail = document.getElementById("heartbeat-detail");
    if (detail) detail.textContent = details[state] || details.active;
    var elapsed = document.getElementById("heartbeat-elapsed");
    if (elapsed) elapsed.textContent = formatDuration(data.elapsed_seconds);
    var stageAge = document.getElementById("heartbeat-stage-age");
    if (stageAge) stageAge.textContent = formatDuration(data.stage_age_seconds);
    var heartbeatAge = document.getElementById("heartbeat-age");
    if (heartbeatAge) heartbeatAge.textContent = formatDuration(data.heartbeat_age_seconds);
    var sidebar = document.getElementById("sidebar-heartbeat");
    if (sidebar) {
      sidebar.textContent = (titles[state] || titles.active) + " · " + formatDuration(data.elapsed_seconds) + " elapsed";
    }
  }

  function showReconnecting(visible, message) {
    var notice = document.getElementById("poll-reconnecting");
    if (!notice) return;
    if (message) notice.textContent = message;
    notice.classList.toggle("d-none", !visible);
  }

  function scheduleNext() {
    if (_terminal || !_started) return;
    if (_pollTimer !== null) clearTimeout(_pollTimer);
    _pollTimer = setTimeout(poll, _currentInterval);
  }

  function handleTransient(message) {
    showReconnecting(
      true,
      message || "Connection interrupted. The forecast continues in the background; reconnecting automatically…"
    );
    _currentInterval = Math.min(_currentInterval * 2, MAX_BACKOFF_MS);
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

    var progressCard = document.querySelector(".forecast-progress-card");
    if (progressCard) {
      var pageErr = document.getElementById("progress-error-message");
      if (!pageErr) {
        pageErr = document.createElement("div");
        pageErr.id = "progress-error-message";
        pageErr.className = "alert alert-danger mt-4 text-start";
        pageErr.setAttribute("role", "alert");
        progressCard.appendChild(pageErr);
      }
      pageErr.textContent = message;

      var progressStep = document.getElementById("progress-step");
      if (progressStep) progressStep.textContent = "Forecast stopped.";

      var retryLink = document.getElementById("progress-retry-link");
      if (!retryLink) {
        retryLink = document.createElement("a");
        retryLink.id = "progress-retry-link";
        retryLink.className = "btn btn-primary mt-3";
        retryLink.href = "/forecast-setup";
        retryLink.textContent = "Back to setup";
        progressCard.appendChild(retryLink);
      }
      return;
    }

    // The dedicated progress screen handles errors above. If this ever runs
    // without that markup, fall back to setup where the server displays it.
    if (window.location.pathname === "/forecast-progress") {
      window.location.assign("/forecast-setup");
      return;
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
    // A status request can complete the job and clear its session ID. Never
    // let a second, stale request replace the successful report redirect.
    if (_pollInFlight || _terminal) return;
    _pollInFlight = true;
    fetch("/api/jobs/status", {
      headers: { "X-CSRFToken": "" },
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        if (_terminal) return;
        var data = result.data;
        if (data.transient || !result.ok) {
          handleTransient(data.error);
          return;
        }
        showReconnecting(false);
        _currentInterval = POLL_INTERVAL_MS;
        if (data.error) {
          _terminal = true;
          stop();
          showError(data.error);
          return;
        }

        updateProgress(data.progress || 0, data.step || "Processing...");
        updateHeartbeat(data);

        if (data.done) {
          _terminal = true;
          stop();
          if (data.redirect) {
            window.location.assign(data.redirect);
          } else {
            window.location.reload();
          }
        } else if (data.status === "error") {
          _terminal = true;
          stop();
          showError(data.error || "Analysis failed.");
        }
      })
      .catch(function (err) {
        if (_terminal) return;
        handleTransient("Connection interrupted. The forecast continues in the background; reconnecting automatically…");
      })
      .finally(function () {
        _pollInFlight = false;
        scheduleNext();
      });
  }

  /** Start resilient polling. */
  function start() {
    if (_started) return;
    _started = true;
    _terminal = false;
    _currentInterval = POLL_INTERVAL_MS;
    poll();
    var progressArea = document.getElementById("progress-area");
    if (progressArea) progressArea.classList.remove("d-none");
  }

  /** Stop the polling interval. */
  function stop() {
    _started = false;
    if (_pollTimer !== null) {
      clearTimeout(_pollTimer);
      _pollTimer = null;
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
