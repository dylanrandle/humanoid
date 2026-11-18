import assert from "node:assert/strict";
import test from "node:test";
import { autonomyAvailability } from "../../../src/humanoid/ui/static/js/autonomy.js";

test("autonomy cannot start until the runner and task are ready", () => {
  const snapshot = { runtime: "sim", processes: { stack: { running: false } } };
  const task = { task: "toy_car_cleanup", ready: false, running: false };
  assert.equal(autonomyAvailability(snapshot, task, false).start, false);
  assert.equal(autonomyAvailability(snapshot, { ...task, ready: true }, false).start, true);
  assert.equal(autonomyAvailability({ ...snapshot, runtime: "real" }, { ...task, ready: true }, false).start, false);
});

test("stop stays available while an operation is pending", () => {
  assert.equal(autonomyAvailability(null, { running: true }, true).stop, true);
});
