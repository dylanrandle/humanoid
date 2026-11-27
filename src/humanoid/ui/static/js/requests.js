import {
  API,
  ApiKey,
  ProcessAction,
  ProcessName,
  Runtime,
} from "./constants.js";

const REAL_STACK_CONFIRMATION =
  "Start the real robot stack now? Confirm the robot workspace is clear.";
const REAL_REPLAY_CONFIRMATION =
  "Replay this recording on real hardware? Recorded commands will move the robot.";

export function safetyPayload(snapshot, realHardwareAcknowledged = false) {
  return {
    [ApiKey.EXPECTED_RUNTIME]: snapshot.runtime,
    [ApiKey.EXPECTED_ROBOT]: snapshot.robot,
    [ApiKey.REAL_HARDWARE_ACKNOWLEDGED]: realHardwareAcknowledged,
  };
}

export function runtimeRequest(snapshot, runtime) {
  if (!snapshot || runtime === snapshot.runtime) return null;
  return {
    path: API.RUNTIME,
    payload: {
      [ApiKey.RUNTIME]: runtime,
      ...safetyPayload(snapshot),
    },
  };
}

export function robotRequest(snapshot, robot) {
  if (!snapshot || robot === snapshot.robot) return null;
  return {
    path: API.ROBOT,
    payload: { [ApiKey.ROBOT]: robot, ...safetyPayload(snapshot) },
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

export function recordingRequest(action) {
  return { path: API.recording(action), payload: {} };
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
      [ApiKey.RECORDING]: recordingId,
      ...safetyPayload(snapshot, replayingOnHardware),
    },
  };
}
