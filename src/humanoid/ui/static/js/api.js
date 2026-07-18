import { API } from "./constants.js";

export async function fetchStatus() {
  const response = await fetch(API.STATUS, { cache: "no-store" });
  const result = await readJson(response);
  if (!response.ok || result === null) {
    throw new Error(result?.error || "Operator console is unavailable.");
  }
  return result;
}

export async function post(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await readJson(response);
  if (!response.ok || !result?.ok) {
    throw new Error(result?.error || `Request failed (${response.status})`);
  }
  return result.status;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}
