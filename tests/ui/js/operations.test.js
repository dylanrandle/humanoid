import assert from "node:assert/strict";
import test from "node:test";

import { performOperation } from "../../../src/humanoid/ui/static/js/operations.js";

function operationOptions(state, action) {
  return {
    state,
    key: "stack",
    action,
    renderSnapshot() {},
    showError() {},
    async refresh() {},
  };
}

test("busy operations reject duplicate actions", async () => {
  let finish;
  let calls = 0;
  const state = { snapshot: null, busy: new Set(), refreshPromise: null };
  const action = () => {
    calls += 1;
    return new Promise((resolve) => { finish = resolve; });
  };

  const first = performOperation(operationOptions(state, action));
  const duplicate = await performOperation(operationOptions(state, action));

  assert.equal(duplicate, false);
  assert.equal(calls, 1);
  finish({ processes: {} });
  await first;
  assert.equal(state.busy.size, 0);
});

test("failed operations surface their error, unlock controls, and refresh", async () => {
  const errors = [];
  let refreshes = 0;
  const state = { snapshot: { original: true }, busy: new Set(), refreshPromise: null };
  const options = operationOptions(state, async () => { throw new Error("process failed"); });
  options.showError = (message) => errors.push(message);
  options.refresh = async () => { refreshes += 1; };

  await performOperation(options);

  assert.deepEqual(errors, ["process failed"]);
  assert.equal(state.busy.size, 0);
  assert.equal(refreshes, 1);
});

test("completed operation unlocks its button during a slow background poll", async () => {
  let finishPoll;
  const state = {
    snapshot: null,
    busy: new Set(),
    refreshPromise: new Promise((resolve) => { finishPoll = resolve; }),
  };
  const options = operationOptions(state, async () => ({ mode: "idle" }));
  const renders = [];
  options.renderSnapshot = () => renders.push([state.snapshot?.mode, state.busy.has("stack")]);
  const result = performOperation(options);
  await Promise.resolve();
  assert.deepEqual(renders, [[undefined, true], ["idle", false]]);
  finishPoll();
  await result;
});
