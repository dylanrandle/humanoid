import assert from "node:assert/strict";
import test from "node:test";

import { fetchLogs } from "../../../src/humanoid/ui/static/js/api.js";

test("log requests fetch only entries after the current cursor", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  const expected = {
    cursor: 8,
    entries: [{ cursor: 8, message: "[INFO] Stack started" }],
    reset: false,
    capacity: 200,
  };
  globalThis.fetch = async (...args) => {
    requests.push(args);
    return { ok: true, json: async () => expected };
  };

  try {
    assert.deepEqual(await fetchLogs(7), expected);
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requests[0][0], "/api/logs?after=7");
  assert.deepEqual(requests[0][1], { cache: "no-store" });
});
