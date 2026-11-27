import { API, ApiKey } from "./constants.js";

export async function fetchStatus(path = API.STATUS) {
  const response = await fetch(path, { cache: "no-store" });
  const result = await readJson(response);
  if (!response.ok || result === null) {
    throw new Error(result?.error || "Operator console is unavailable.");
  }
  return result;
}

export async function fetchLogs(after = null) {
  const query =
    after === null ? "" : `?${new URLSearchParams({ [ApiKey.AFTER]: after })}`;
  const response = await fetch(`${API.LOGS}${query}`, {
    cache: "no-store",
  });
  const result = await readJson(response);
  if (!response.ok || result === null) {
    throw new Error(result?.error || "Logs are unavailable.");
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
