import { els } from "./dom.js";

let toastTimer = null;

export function showToast(message) {
  window.clearTimeout(toastTimer);
  els.toastMessage.textContent = message;
  els.toast.hidden = false;
  toastTimer = window.setTimeout(() => { els.toast.hidden = true; }, 5000);
}
