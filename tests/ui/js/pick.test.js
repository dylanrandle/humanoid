import assert from "node:assert/strict";
import test from "node:test";
import { pickRequest, renderPick } from "../../../src/humanoid/ui/static/js/pick.js";

function snapshot() {
  return {
    runtime: "sim", robot: "triskel",
    processes: { stack: { running: true }, pick: { running: true } },
    orchestrator: { connected: true, mode: "idle" },
    pick: { models_ready: false, busy: false, message: "Loading models" },
  };
}

function elements() {
  return Object.fromEntries(["process", "object", "run", "cancel", "status"].map(name => [name, {value: "red toy car"}]));
}

test("run waits for models, text and idle; cancel remains available during work", () => {
  const state = snapshot(), els = elements();
  renderPick(state, new Set(), els);
  assert.equal(els.run.disabled, true);
  state.pick.models_ready = true;
  renderPick(state, new Set(), els);
  assert.equal(els.run.disabled, false);
  state.pick.busy = true;
  state.orchestrator.mode = "system";
  renderPick(state, new Set(), els);
  assert.equal(els.run.disabled, true);
  assert.equal(els.cancel.disabled, false);
  assert.equal(els.process.disabled, false);
});

test("hardware motion requires explicit acknowledgement and sends the selected configuration", () => {
  const state = snapshot(); state.runtime = "real";
  assert.equal(pickRequest(state, "car", () => false), null);
  const request = pickRequest(state, "  red toy car  ", () => true);
  assert.equal(request.payload.object, "red toy car");
  assert.equal(request.payload.expected_robot, "triskel");
  assert.equal(request.payload.real_hardware_acknowledged, true);
});

test("empty object cannot create a request", () => {
  assert.equal(pickRequest(snapshot(), "  ", () => true), null);
});
