import assert from "node:assert/strict";
import test from "node:test";

import { ProcessName, Runtime } from "../../../src/humanoid/ui/static/js/constants.js";
import {
  collectionRequest,
  collectionSceneRequest,
  collectionUnavailable,
  loggingRequest,
  processRequest,
  replayRequest,
  robotRequest,
  runtimeRequest,
} from "../../../src/humanoid/ui/static/js/requests.js";

test("collection uses the existing stack and stops only the observed session", () => {
  const current = snapshot();
  const settings = { task: "cleanup", episodes: 2, max_attempts: 6, seed: 9, max_sim_seconds: 120, output: "logs/recordings" };
  const start = collectionRequest(current, settings);
  assert.equal(start.path, "/api/collection/start");
  assert.deepEqual(start.payload.settings, settings);
  assert.equal(start.payload.expected_runtime, "sim");
  current.collection = {state: "running", run_id: "my-run"};
  assert.deepEqual(collectionRequest(current, settings), {path: "/api/collection/stop", payload: {run_id: "my-run"}});
});

test("task scene loading uses selection and waits for an idle compatible stack", () => {
  const current = snapshot();
  current.robot = "triskel";
  current.processes.stack.running = true;
  current.orchestrator = {connected: true, mode: "idle"};
  current.logging = {state: "stopped"};
  current.collection_tasks = [{name: "cleanup", robot: "triskel"}, {name: "other", robot: "triskel"}];
  current.collection_defaults = {task: "cleanup", seed: 0, episodes: 1, max_attempts: 3, max_sim_seconds: 600, output: "logs/recordings"};
  const request = collectionSceneRequest(current, "other", 42);
  assert.equal(request.path, "/api/collection/prepare");
  assert.equal(request.payload.settings.task, "other");
  assert.equal(request.payload.settings.seed, 42);
  for (const change of [
    {runtime: "real"},
    {robot: "panda"},
    {orchestrator: {connected: true, mode: "keyboard"}},
    {logging: {state: "running"}},
    {collection: {state: "running"}},
    {collection: {state: "preparing"}},
  ]) assert.equal(collectionSceneRequest({...current, ...change}, "other", 42), null);
  assert.equal(collectionSceneRequest(current, "missing", 42), null);
  assert.equal(collectionSceneRequest(current, "other", NaN), null);
});

test("logging controls use the dedicated start and stop endpoints", () => {
  assert.deepEqual(loggingRequest("start"), { path: "/api/logging/start", payload: {} });
  assert.deepEqual(loggingRequest("stop"), { path: "/api/logging/stop", payload: {} });
});

test("empty scene can load for every robot without allowing collection", () => {
  const current = snapshot();
  current.processes.stack.running = true;
  current.orchestrator = {connected: true, mode: "idle"};
  current.logging = {state: "stopped"};
  current.collection_tasks = [{name: "empty", robot: null, collectable: false}];
  current.collection_defaults = {task: "empty", seed: 0};
  for (const robot of ["panda", "so101", "triskel"]) {
    current.robot = robot;
    const request = collectionSceneRequest(current, "empty", 0);
    assert.equal(request.path, "/api/collection/prepare");
    assert.equal(request.payload.settings.task, "empty");
    assert.equal(request.payload.expected_robot, robot);
    assert.match(collectionUnavailable(current, "empty"), /teleoperation or manual recording/);
  }
});

function snapshot(runtime = Runtime.SIM) {
  return {
    runtime,
    robot: "panda",
    processes: {
      stack: { running: false },
      replay: { running: false },
      keyboard: { running: false },
      oculus: { running: false },
    },
    replay: {
      running: false,
      file_name: null,
      exit_code: null,
      last_output: null,
    },
  };
}

test("replay selects a server-managed recording and stops through JSON", () => {
  const current = snapshot();

  assert.equal(replayRequest(current, "", () => true), null);
  assert.deepEqual(replayRequest(current, "recording_1", () => true), {
    path: "/api/replay/start",
    payload: {
      recording: "recording_1",
      expected_runtime: "sim",
      expected_robot: "panda",
      real_hardware_acknowledged: false,
    },
  });

  current.replay.running = true;
  assert.deepEqual(replayRequest(current, "recording_1", () => true), {
    path: "/api/replay/stop",
    payload: {},
  });
});

test("real replay requires confirmation and carries hardware acknowledgement", () => {
  const current = snapshot(Runtime.REAL);

  assert.equal(replayRequest(current, "recording_1", () => false), null);
  assert.deepEqual(replayRequest(current, "recording_1", () => true), {
    path: "/api/replay/start",
    payload: {
      recording: "recording_1",
      expected_runtime: "real",
      expected_robot: "panda",
      real_hardware_acknowledged: true,
    },
  });
});

test("real runtime selection does not require hardware confirmation", () => {
  const current = snapshot();
  let confirmations = 0;

  const request = runtimeRequest(current, Runtime.REAL, () => { confirmations += 1; });
  assert.deepEqual(request, {
    path: "/api/runtime",
    payload: {
      runtime: "real",
      expected_runtime: "sim",
      expected_robot: "panda",
      real_hardware_acknowledged: false,
    },
  });
  assert.equal(confirmations, 0);
});

test("real stack start requires confirmation and carries the observed configuration", () => {
  const current = snapshot(Runtime.REAL);
  assert.equal(processRequest(current, ProcessName.STACK, () => false), null);

  const request = processRequest(current, ProcessName.STACK, () => true);
  assert.deepEqual(request, {
    path: "/api/processes/stack/start",
    payload: {
      expected_runtime: "real",
      expected_robot: "panda",
      real_hardware_acknowledged: true,
    },
  });
});

test("process stop is always available without hardware acknowledgement", () => {
  const current = snapshot(Runtime.REAL);
  current.processes.stack.running = true;
  let confirmations = 0;

  const request = processRequest(current, ProcessName.STACK, () => { confirmations += 1; });

  assert.deepEqual(request, { path: "/api/processes/stack/stop", payload: {} });
  assert.equal(confirmations, 0);
});

test("teleop start and robot selection carry stale-configuration guards", () => {
  const current = snapshot();

  assert.deepEqual(processRequest(current, ProcessName.KEYBOARD, () => true), {
    path: "/api/processes/keyboard/start",
    payload: {
      expected_runtime: "sim",
      expected_robot: "panda",
      real_hardware_acknowledged: false,
    },
  });
  assert.deepEqual(robotRequest(current, "so101"), {
    path: "/api/robot",
    payload: {
      robot: "so101",
      expected_runtime: "sim",
      expected_robot: "panda",
      real_hardware_acknowledged: false,
    },
  });
});
