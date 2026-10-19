"use strict";

(() => {
  const form = document.getElementById("search-form");
  const input = document.getElementById("search");
  const results = document.getElementById("results");
  const status = document.getElementById("search-status");
  if (!form || !input) return;

  const submitButton = form.querySelector(".search-button");
  const submitLabel = form.querySelector(".button-label");

  function resetLoading() {
    form.classList.remove("is-loading");
    form.removeAttribute("aria-busy");
    if (results) results.setAttribute("aria-busy", "false");
    if (submitButton) submitButton.disabled = false;
    if (submitLabel) submitLabel.textContent = "Search";
    if (status) status.textContent = "";
  }

  input.addEventListener("input", () => input.setCustomValidity(""));

  form.addEventListener("submit", (event) => {
    if (!input.value.trim()) {
      event.preventDefault();
      input.setCustomValidity("Enter a movie description or a few keywords.");
      input.reportValidity();
      return;
    }
    input.setCustomValidity("");
    form.classList.add("is-loading");
    form.setAttribute("aria-busy", "true");
    if (results) results.setAttribute("aria-busy", "true");
    if (submitButton) submitButton.disabled = true;
    if (submitLabel) submitLabel.textContent = "Searching…";
    if (status) status.textContent = "Searching movie plots by keyword, meaning, and combined rank.";
  });

  document.querySelectorAll(".example-query[data-query]").forEach((example) => {
    example.addEventListener("click", (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof form.requestSubmit !== "function") return;
      event.preventDefault();
      input.value = example.dataset.query;
      input.setCustomValidity("");
      form.requestSubmit();
    });
  });

  // A browser back/forward-cache restore can preserve a disabled submit button.
  window.addEventListener("pageshow", resetLoading);
})();
