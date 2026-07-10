/**
 * Chat interface module.
 *
 * Handles sending messages to ``/api/chat``, appending them to the chat
 * history div, and rendering any Plotly visualisations returned by the
 * backend.
 */

(function () {
  "use strict";

  var ENTER_KEY = 13;

  /** Read the CSRF token from the page meta tag. */
  function getCsrfToken() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  /**
   * Append a message bubble to the chat messages container.
   *
   * @param {string}  role     ``'user'`` or ``'assistant'``.
   * @param {string}  content  Plain text content.
   * @param {object|null} vizData  Optional visualisation data object.
   * @param {string|null} vizType  ``'pie'``, ``'dynamic'``, or null.
   */
  function appendMessage(role, content, vizData, vizType) {
    var container = document.getElementById("chat-messages");
    if (!container) return;

    var wrapper = document.createElement("div");
    wrapper.className = "chat-message " + role;

    var bubble = document.createElement("div");
    bubble.className = "chat-bubble";

    var contentEl = document.createElement("div");
    contentEl.className = "chat-content";
    if (role === "assistant" && typeof marked !== "undefined") {
      contentEl.innerHTML = DOMPurify.sanitize(marked.parse(content));
    } else {
      contentEl.textContent = content;
    }
    bubble.appendChild(contentEl);

    if (vizData) {
      var chartEl = document.createElement("div");
      chartEl.className = "chart-container mt-2";
      chartEl.style.minHeight = "200px";
      chartEl.dataset.chart = JSON.stringify(vizData);
      chartEl.dataset.chartType = vizType || "dynamic";
      bubble.appendChild(chartEl);
      wrapper.appendChild(bubble);
      container.appendChild(wrapper);

      if (vizType === "pie") {
        Charts.renderPie(chartEl, vizData);
      } else {
        Charts.renderAll();
      }
    } else {
      wrapper.appendChild(bubble);
      container.appendChild(wrapper);
    }

    container.scrollTop = container.scrollHeight;
  }

  /** Send the current chat input value to the backend. */
  function send() {
    var input = document.getElementById("chat-input");
    var spinner = document.getElementById("chat-spinner");
    var sendBtn = document.getElementById("chat-send");
    if (!input) return;

    var query = input.value.trim();
    if (!query) return;

    input.value = "";
    appendMessage("user", query, null, null);

    if (spinner) spinner.classList.remove("d-none");
    if (sendBtn) sendBtn.disabled = true;

    fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCsrfToken(),
      },
      body: JSON.stringify({ query: query }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          appendMessage("assistant", "Error: " + data.error, null, null);
          return;
        }

        var vizData = null;
        var vizType = null;

        if (data.visualization_type === "pie" && data.visualization_data) {
          vizData = data.visualization_data;
          vizType = "pie";
        } else if (data.visualization_data) {
          vizData = data.visualization_data;
          vizType = "dynamic";
        } else {
          var parsed = tryParseVizFromText(data.answer || "");
          if (parsed) {
            vizData = parsed;
            vizType = "dynamic";
          }
        }

        appendMessage("assistant", data.answer || "", vizData, vizType);
      })
      .catch(function (err) {
        appendMessage("assistant", "Connection error: " + err, null, null);
      })
      .finally(function () {
        if (spinner) spinner.classList.add("d-none");
        if (sendBtn) sendBtn.disabled = false;
      });
  }

  /**
   * Attempt to extract a JSON visualisation config from the assistant's
   * answer text.
   *
   * @param {string} text  Raw answer string.
   * @returns {object|null}
   */
  function tryParseVizFromText(text) {
    var match = text.match(/```json\s*([\s\S]*?)```/);
    if (!match) {
      match = text.match(/\{[\s\S]*"chart_type"[\s\S]*\}/);
      if (match) {
        try { return JSON.parse(match[0]); } catch (e) { return null; }
      }
      return null;
    }
    try { return JSON.parse(match[1]); } catch (e) { return null; }
  }

  /** Wire up the Enter key on the chat input and scroll-to-bottom button. */
  function init() {
    var input = document.getElementById("chat-input");
    if (!input) return;
    input.addEventListener("keydown", function (e) {
      if (e.keyCode === ENTER_KEY && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    var container = document.getElementById("chat-messages");
    if (container) {
      container.scrollTop = container.scrollHeight;

      // Scroll-to-bottom button logic
      var scrollBtn = document.getElementById("chat-scroll-btn");
      if (scrollBtn) {
        container.addEventListener("scroll", function () {
          var atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 60;
          if (atBottom) {
            scrollBtn.classList.remove("visible");
          } else {
            scrollBtn.classList.add("visible");
          }
        });
        scrollBtn.addEventListener("click", function () {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: "smooth",
          });
        });
      }
    }

    // Example question chips
    var chips = document.querySelectorAll(".chat-example-chip");
    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        input.value = chip.textContent.trim();
        input.focus();
      });
    });
  }

  window.Chat = { send: send, init: init };
}());
