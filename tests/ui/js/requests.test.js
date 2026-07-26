import assert from "node:assert/strict";
import test from "node:test";

import { ProcessName, Runtime } from "../../../src/humanoid/ui/static/js/constants.js";
import {
  loggingRequest,
  processRequest,
  replayRequest,
  robotRequest,
  runtimeRequest,
  sceneRequest,
} from "../../../src/humanoid/ui/static/js/requests.js";

test("logging controls use the dedicated start and stop endpoints", () => {
  assert.deepEqual(loggingRequest("start"), { path: "/api/logging/start", payload: {} });
  assert.deepEqual(loggingRequest("stop"), { path: "/api/logging/stop", payload: {} });
});

function snapshot(runtime = Runtime.SIM) {
  return {
    runtime,
    robot: "panda",
    scene: "empty",
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
      expected_scene: "empty",
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
      expected_scene: "empty",
      real_hardware_acknowledged: true,
    },
  });
});

test("real runtime selection requires confirmation and carries acknowledgement", () => {
  const current = snapshot();
  assert.equal(runtimeRequest(current, Runtime.REAL, () => false), null);

  const request = runtimeRequest(current, Runtime.REAL, () => true);
  assert.deepEqual(request, {
    path: "/api/runtime",
    payload: {
      runtime: "real",
      expected_runtime: "sim",
      expected_robot: "panda",
      expected_scene: "empty",
      real_hardware_acknowledged: true,
    },
  });
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
      expected_scene: "empty",
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
      expected_scene: "empty",
      real_hardware_acknowledged: false,
    },
  });
  assert.deepEqual(robotRequest(current, "so101"), {
    path: "/api/robot",
    payload: {
      robot: "so101",
      expected_runtime: "sim",
      expected_robot: "panda",
      expected_scene: "empty",
      real_hardware_acknowledged: false,
    },
  });
});

test("scene selection carries stale-configuration guards", () => {
  const current = snapshot();

  assert.equal(sceneRequest(current, "empty"), null);
  assert.deepEqual(sceneRequest(current, "floor-and-cube"), {
    path: "/api/scene",
    payload: {
      scene: "floor-and-cube",
      expected_runtime: "sim",
      expected_robot: "panda",
      expected_scene: "empty",
      real_hardware_acknowledged: false,
    },
  });
});
