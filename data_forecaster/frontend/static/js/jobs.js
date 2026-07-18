/**
 * Job queue polling and rendering for the per-user forecast job queue page.
 *
 * ES5-compatible IIFE module.  Polls /api/jobs/mine at a 2.5s interval,
 * pauses when the page is hidden, backs off on network failures, and
 * renders the table rows in-place.  Cancel and finalize actions use the
 * existing CSRF convention (X-CSRFToken header from <meta name="csrf-token">).
 */
(function () {
    "use strict";

    var POLL_INTERVAL_MS = 2500;
    var MAX_BACKOFF_MS = 10000;
    var currentInterval = POLL_INTERVAL_MS;
    var pollTimer = null;
    var pollInFlight = false;
    var stopped = false;

    function getCsrfToken() {
        var meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute("content") : "";
    }

    function fetchJobs() {
        if (pollInFlight || stopped || document.hidden) return;
        pollInFlight = true;
        fetch("/api/jobs/mine", { credentials: "same-origin" })
            .then(function (resp) {
                if (!resp.ok) throw new Error("HTTP " + resp.status);
                return resp.json();
            })
            .then(function (jobs) {
                setConnectionStatus(false);
                renderJobs(jobs);
                maybeContinuePolling(jobs);
            })
            .catch(function () {
                setConnectionStatus(true);
                currentInterval = Math.min(currentInterval * 2, MAX_BACKOFF_MS);
                scheduleNext();
            })
            .then(function () {
                pollInFlight = false;
            });
    }

    function maybeContinuePolling(jobs) {
        var hasActive = false;
        var hasRetryableFinalization = false;
        for (var i = 0; i < jobs.length; i++) {
            var s = jobs[i].status;
            if (s === "pending" || s === "running" || s === "cancelling") {
                hasActive = true;
            }
            if (
                s === "done" &&
                !jobs[i].report_ready &&
                jobs[i].finalization_error !== "report_limit"
            ) {
                hasRetryableFinalization = true;
            }
        }
        if (!hasActive && !hasRetryableFinalization) {
            stop();
            return;
        }
        if (hasRetryableFinalization && !hasActive) {
            currentInterval = Math.min(currentInterval * 2, MAX_BACKOFF_MS);
        } else {
            currentInterval = POLL_INTERVAL_MS;
        }
        scheduleNext();
    }

    function scheduleNext() {
        if (stopped || document.hidden) return;
        if (pollTimer) clearTimeout(pollTimer);
        pollTimer = setTimeout(fetchJobs, currentInterval);
    }

    function stop() {
        stopped = true;
        if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
        }
    }

    function start() {
        stopped = false;
        currentInterval = POLL_INTERVAL_MS;
        fetchJobs();
    }

    function refreshAfterAction() {
        if (stopped) {
            start();
        } else {
            fetchJobs();
        }
    }

    function statusBadgeClass(status) {
        switch (status) {
            case "running":
                return "text-bg-primary";
            case "cancelling":
                return "text-bg-warning";
            case "done":
                return "text-bg-success";
            case "error":
                return "text-bg-danger";
            case "cancelled":
                return "text-bg-secondary";
            default:
                return "text-bg-secondary";
        }
    }

    function setConnectionStatus(visible) {
        var notice = document.getElementById("jobs-connection-status");
        if (notice) notice.classList.toggle("d-none", !visible);
    }

    function formatDuration(value) {
        if (value === null || typeof value === "undefined") return "—";
        var seconds = Math.max(0, parseInt(value, 10) || 0);
        if (seconds < 60) return seconds + "s";
        if (seconds < 3600) return Math.floor(seconds / 60) + "m " + (seconds % 60) + "s";
        return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
    }

    function livenessLabel(job) {
        switch (job.liveness) {
            case "active": return "Worker active";
            case "delayed": return "Worker signal delayed";
            case "stale": return "Worker status uncertain";
            case "terminal": return "Finished";
            default: return "Queued";
        }
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function renderJobs(jobs) {
        var tbody = document.getElementById("jobs-tbody");
        var empty = document.getElementById("jobs-empty");
        var wrapper = document.getElementById("jobs-table-wrapper");
        var clearButton = document.getElementById("clear-terminal-jobs");
        if (!tbody || !empty || !wrapper) return;

        if (!jobs || jobs.length === 0) {
            empty.classList.remove("d-none");
            wrapper.classList.add("d-none");
            if (clearButton) clearButton.disabled = true;
            return;
        }
        empty.classList.add("d-none");
        wrapper.classList.remove("d-none");

        var html = "";
        var hasTerminalJobs = false;
        for (var i = 0; i < jobs.length; i++) {
            var job = jobs[i];
            if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
                hasTerminalJobs = true;
            }
            var name = escapeHtml(job.report_name || "Untitled forecast");
            var status = escapeHtml(job.status);
            var badgeClass = statusBadgeClass(job.status);
            var progress = parseInt(job.progress, 10) || 0;
            var step = escapeHtml(job.step || "");
            var liveness = escapeHtml(job.liveness || "queued");
            var activity = '<div class="small text-secondary"><span class="job-liveness-dot job-liveness-' +
                liveness + '"></span> ' + escapeHtml(livenessLabel(job)) + " · " +
                escapeHtml(formatDuration(job.elapsed_seconds)) + " elapsed";
            if (job.liveness === "active" || job.liveness === "delayed" || job.liveness === "stale") {
                activity += " · stage " + escapeHtml(formatDuration(job.stage_age_seconds));
            }
            activity += "</div>";
            var error = job.error ? '<div class="small text-danger">' + escapeHtml(job.error) + "</div>" : "";
            var submitted = escapeHtml(job.queued_at || "");

            var actions = "";
            if (job.report_ready && job.report_id) {
                actions += '<a class="btn btn-outline-light btn-sm" href="/reports/' +
                    parseInt(job.report_id, 10) + '">View</a> ';
            }
            if (job.status === "done" && !job.report_ready) {
                actions += '<button class="btn btn-outline-primary btn-sm job-finalize-btn" data-job-id="' +
                    escapeHtml(job.job_id) + '" type="button">Finalize</button> ';
            }
            if (job.finalization_error === "report_limit") {
                actions += '<span class="badge text-bg-warning">Report limit reached</span> ';
                actions += '<a class="btn btn-outline-secondary btn-sm" href="/reports">Manage reports</a> ';
            } else if (job.finalization_error) {
                actions += '<span class="badge text-bg-warning">Finalization pending</span> ';
            }
            if (job.can_cancel) {
                actions += '<button class="btn btn-outline-danger btn-sm job-cancel-btn" data-job-id="' +
                    escapeHtml(job.job_id) + '" type="button">Cancel</button>';
            }

            html += '<tr data-job-id="' + escapeHtml(job.job_id) + '">' +
                "<td>" + name + "</td>" +
                '<td><span class="badge ' + badgeClass + '">' + status + "</span>" +
                '<div class="small text-secondary">' + step + "</div>" + activity + error + "</td>" +
                '<td><div class="progress" style="height:8px;">' +
                '<div class="progress-bar" style="width:' + progress + '%"></div>' +
                '</div><small class="text-secondary">' + progress + "%</small></td>" +
                '<td class="text-secondary">' + submitted + "</td>" +
                '<td class="text-end">' + actions + "</td>" +
                "</tr>";
        }
        tbody.innerHTML = html;
        if (clearButton) clearButton.disabled = !hasTerminalJobs;
        attachActionHandlers();
    }

    function attachActionHandlers() {
        var cancelBtns = document.querySelectorAll(".job-cancel-btn");
        for (var i = 0; i < cancelBtns.length; i++) {
            cancelBtns[i].addEventListener("click", handleCancel);
        }
        var finalizeBtns = document.querySelectorAll(".job-finalize-btn");
        for (var j = 0; j < finalizeBtns.length; j++) {
            finalizeBtns[j].addEventListener("click", handleFinalize);
        }
    }

    function handleCancel(event) {
        var btn = event.target;
        var jobId = btn.getAttribute("data-job-id");
        btn.disabled = true;
        btn.textContent = "Cancelling…";
        fetch("/api/jobs/" + encodeURIComponent(jobId) + "/cancel", {
            method: "POST",
            headers: { "X-CSRFToken": getCsrfToken() },
            credentials: "same-origin",
        })
            .then(function (resp) {
                return resp.json().then(function (body) {
                    return { status: resp.status, body: body };
                });
            })
            .then(function (result) {
                if (result.status === 200) {
                    refreshAfterAction();
                } else {
                    btn.disabled = false;
                    btn.textContent = "Cancel";
                    alert(result.body.error || "Failed to cancel job.");
                }
            })
            .catch(function () {
                btn.disabled = false;
                btn.textContent = "Cancel";
                alert("Network error while cancelling job.");
            });
    }

    function handleFinalize(event) {
        var btn = event.target;
        var jobId = btn.getAttribute("data-job-id");
        btn.disabled = true;
        btn.textContent = "Finalizing…";
        fetch("/api/jobs/" + encodeURIComponent(jobId) + "/finalize", {
            method: "POST",
            headers: { "X-CSRFToken": getCsrfToken() },
            credentials: "same-origin",
        })
            .then(function (resp) {
                return resp.json().then(function (body) {
                    return { status: resp.status, body: body };
                });
            })
            .then(function (result) {
                if (result.status === 200) {
                    refreshAfterAction();
                } else {
                    btn.disabled = false;
                    btn.textContent = "Finalize";
                    alert(result.body.error || "Failed to finalize report.");
                }
            })
            .catch(function () {
                btn.disabled = false;
                btn.textContent = "Finalize";
                alert("Network error while finalizing report.");
            });
    }

    function handleClearTerminal() {
        var btn = document.getElementById("clear-terminal-jobs");
        if (!btn || btn.disabled) return;
        if (!window.confirm("Clear all of your completed, failed, and cancelled jobs? Saved reports will not be deleted.")) return;
        btn.disabled = true;
        btn.textContent = "Clearing…";
        fetch("/api/jobs/mine/terminal", {
            method: "POST",
            headers: { "X-CSRFToken": getCsrfToken() },
            credentials: "same-origin",
        })
            .then(function (resp) {
                return resp.json().then(function (body) {
                    return { status: resp.status, body: body };
                });
            })
            .then(function (result) {
                if (result.status === 200) {
                    btn.textContent = "Clear completed/failed";
                    refreshAfterAction();
                } else {
                    btn.disabled = false;
                    btn.textContent = "Clear completed/failed";
                    alert(result.body.error || "Failed to clear completed jobs.");
                }
            })
            .catch(function () {
                btn.disabled = false;
                btn.textContent = "Clear completed/failed";
                alert("Network error while clearing completed jobs.");
            });
    }

    // Pause polling when the page is hidden; resume on visibility.
    document.addEventListener("visibilitychange", function () {
        if (document.hidden) {
            if (pollTimer) {
                clearTimeout(pollTimer);
                pollTimer = null;
            }
        } else if (!stopped && !pollInFlight) {
            currentInterval = POLL_INTERVAL_MS;
            fetchJobs();
        }
    });

    // Auto-start on the jobs page.
    if (document.body && document.body.getAttribute("data-page") === "jobs") {
        start();
    }

    var clearTerminalButton = document.getElementById("clear-terminal-jobs");
    if (clearTerminalButton) clearTerminalButton.addEventListener("click", handleClearTerminal);

    window.JobQueue = { start: start, stop: stop };
})();
