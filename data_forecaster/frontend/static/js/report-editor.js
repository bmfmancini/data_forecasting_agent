/** Protect unsaved section wording and reset edits when Cancel is selected. */
(function () {
  "use strict";
  var dirty = new Set();
  document.querySelectorAll(".report-edit-form").forEach(function (form) {
    form.addEventListener("input", function () { dirty.add(form); });
    form.querySelector(".report-edit-cancel").addEventListener("click", function () {
      form.reset();
      dirty.delete(form);
      form.closest("details").open = false;
    });
    form.addEventListener("submit", function () { dirty.delete(form); });
  });
  window.addEventListener("beforeunload", function (event) {
    if (dirty.size) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
}());
