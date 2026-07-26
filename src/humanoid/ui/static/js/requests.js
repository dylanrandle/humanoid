import {
  API,
  PayloadKey,
  ProcessAction,
  ProcessName,
  Runtime,
} from "./constants.js";

const REAL_RUNTIME_CONFIRMATION =
  "Switch to the real robot runtime? Starting the stack can command physical hardware.";
const REAL_STACK_CONFIRMATION =
  "Start the real robot stack now? Confirm the robot workspace is clear.";
const REAL_REPLAY_CONFIRMATION =
  "Replay this recording on real hardware? Recorded commands will move the robot.";

export function safetyPayload(snapshot, realHardwareAcknowledged = false) {
  return {
    [PayloadKey.EXPECTED_RUNTIME]: snapshot.runtime,
    [PayloadKey.EXPECTED_ROBOT]: snapshot.robot,
    [PayloadKey.EXPECTED_SCENE]: snapshot.scene,
    [PayloadKey.REAL_HARDWARE_ACKNOWLEDGED]: realHardwareAcknowledged,
  };
}

export function runtimeRequest(snapshot, runtime, confirmHardware) {
  if (!snapshot || runtime === snapshot.runtime) return null;
  const selectingReal = runtime === Runtime.REAL;
  if (selectingReal && !confirmHardware(REAL_RUNTIME_CONFIRMATION)) return null;
  return {
    path: API.RUNTIME,
    payload: {
      [PayloadKey.RUNTIME]: runtime,
      ...safetyPayload(snapshot, selectingReal),
    },
  };
}

export function robotRequest(snapshot, robot) {
  if (!snapshot || robot === snapshot.robot) return null;
  return {
    path: API.ROBOT,
    payload: { [PayloadKey.ROBOT]: robot, ...safetyPayload(snapshot) },
  };
}

export function sceneRequest(snapshot, scene) {
  if (!snapshot || scene === snapshot.scene) return null;
  return {
    path: API.SCENE,
    payload: { [PayloadKey.SCENE]: scene, ...safetyPayload(snapshot) },
  };
}

export function processRequest(snapshot, name, confirmHardware) {
  const process = snapshot?.processes[name];
  if (!process) return null;
  const action = process.running ? ProcessAction.STOP : ProcessAction.START;
  const startingRealStack =
    name === ProcessName.STACK &&
    action === ProcessAction.START &&
    snapshot.runtime === Runtime.REAL;
  if (startingRealStack && !confirmHardware(REAL_STACK_CONFIRMATION))
    return null;
  return {
    path: API.process(name, action),
    payload:
      action === ProcessAction.START
        ? safetyPayload(snapshot, startingRealStack)
        : {},
  };
}

export function loggingRequest(action) {
  return { path: API.logging(action), payload: {} };
}

export function replayRequest(snapshot, recordingId, confirmHardware) {
  if (!snapshot) return null;
  if (snapshot.replay.running) {
    return { path: API.replay(ProcessAction.STOP), payload: {} };
  }
  if (!recordingId) return null;
  const replayingOnHardware = snapshot.runtime === Runtime.REAL;
  if (replayingOnHardware && !confirmHardware(REAL_REPLAY_CONFIRMATION))
    return null;
  return {
    path: API.replay(ProcessAction.START),
    payload: {
      [PayloadKey.RECORDING]: recordingId,
      ...safetyPayload(snapshot, replayingOnHardware),
    },
  };
}
