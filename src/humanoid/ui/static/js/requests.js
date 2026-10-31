import {
  API,
  CollectionAction,
  CollectionRunState,
  LoggingState,
  Mode,
  PayloadKey,
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
    [PayloadKey.EXPECTED_RUNTIME]: snapshot.runtime,
    [PayloadKey.EXPECTED_ROBOT]: snapshot.robot,
    [PayloadKey.REAL_HARDWARE_ACKNOWLEDGED]: realHardwareAcknowledged,
  };
}

export function runtimeRequest(snapshot, runtime) {
  if (!snapshot || runtime === snapshot.runtime) return null;
  return {
    path: API.RUNTIME,
    payload: {
      [PayloadKey.RUNTIME]: runtime,
      ...safetyPayload(snapshot),
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

export function collectionRunning(collection) {
  return [
    CollectionRunState.PREPARING,
    CollectionRunState.STARTING,
    CollectionRunState.RUNNING,
    CollectionRunState.STOPPING,
  ].includes(collection?.state);
}

export function collectionSceneRequest(snapshot, task, seed) {
  if (
    !snapshot?.collection_defaults ||
    collectionRunning(snapshot.collection) ||
    collectionUnavailable(snapshot, task, CollectionAction.PREPARE) ||
    !Number.isSafeInteger(seed) ||
    seed < 0
  )
    return null;
  return {
    path: API.collection(CollectionAction.PREPARE),
    payload: {
      [PayloadKey.SETTINGS]: { ...snapshot.collection_defaults, task, seed },
      ...safetyPayload(snapshot),
    },
  };
}

export function collectionUnavailable(
  snapshot,
  taskName,
  action = CollectionAction.START,
) {
  if (snapshot.runtime !== Runtime.SIM)
    return "Select Simulation to load task scenes or collect episodes.";
  const task = snapshot.collection_tasks?.find(
    (option) => option.name === taskName,
  );
  if (!task) return "No task is available.";
  if (task.robot && task.robot !== snapshot.robot)
    return `Select the ${task.robot} robot for ${task.label}.`;
  if (action === CollectionAction.START && task.collectable === false)
    return "Use teleoperation or manual recording in this scene.";
  if (
    !snapshot.processes[ProcessName.STACK].running ||
    !snapshot.orchestrator.connected
  )
    return "Start the simulation stack and wait for it to be ready.";
  if (snapshot.orchestrator.mode !== Mode.IDLE)
    return "Select Idle before loading a scene or starting collection.";
  if (
    ![LoggingState.STOPPED, LoggingState.FAILED].includes(
      snapshot.logging.state,
    )
  )
    return "Stop manual recording before loading a scene or starting collection.";
  return "";
}

export function collectionRequest(snapshot, settings) {
  if (!snapshot) return null;
  if (collectionRunning(snapshot.collection)) {
    return {
      path: API.collection(CollectionAction.STOP),
      payload: { run_id: snapshot.collection.run_id },
    };
  }
  return {
    path: API.collection(CollectionAction.START),
    payload: { [PayloadKey.SETTINGS]: settings, ...safetyPayload(snapshot) },
  };
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
