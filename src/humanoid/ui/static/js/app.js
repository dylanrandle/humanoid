import { fetchLogs, fetchStatus, post } from "./api.js";
import { setupAgentPanel } from "./agent.js";
import { startCameraStreams } from "./cameras.js";
import {
  API,
  BusyKey,
  OrchestratorParameter,
  ApiKey,
  ProcessName,
} from "./constants.js";
import { els } from "./dom.js";
import { errorMessage, performOperation } from "./operations.js";
import {
  render,
  renderLogs,
  renderLogsDisconnected,
  renderDisconnected,
} from "./render.js";
import {
  recordingRequest,
  processRequest,
  replayRequest,
  robotRequest,
  safetyPayload,
  runtimeRequest,
} from "./requests.js";
import { showToast } from "./toast.js";

const state = {
  snapshot: null,
  busy: new Set(),
  connected: null,
  logs: [],
  logCursor: null,
  logsConnected: null,
  refreshPromise: null,
};
const agent = setupAgentPanel(() => (state.connected ? state.snapshot : null));

function refresh() {
  if (state.refreshPromise) return state.refreshPromise;
  state.refreshPromise = Promise.all([
    loadStatus(),
    loadLogs(),
    agent.refresh(),
  ]).finally(() => {
    state.refreshPromise = null;
  });
  return state.refreshPromise;
}

async function loadStatus() {
  try {
    state.snapshot = await fetchStatus();
    state.connected = true;
    render(state.snapshot, state.busy, els);
  } catch (error) {
    const message = errorMessage(error);
    renderDisconnected(message, els);
    if (state.connected !== false) showToast(message);
    state.connected = false;
  }
  agent.render();
}

async function loadLogs() {
  try {
    const snapshot = await fetchLogs(state.logCursor);
    if (snapshot.reset) state.logs = [];
    state.logs = [
      ...state.logs,
      ...snapshot.entries.map((entry) => entry.message),
    ].slice(-snapshot.capacity);
    state.logCursor = snapshot.cursor;
    state.logsConnected = true;
    renderLogs(state.logs, els);
  } catch (error) {
    renderLogsDisconnected(els);
    if (state.logsConnected !== false) showToast(errorMessage(error));
    state.logsConnected = false;
  }
}

async function perform(key, action) {
  return performOperation({
    state,
    key,
    action,
    renderSnapshot: () => render(state.snapshot, state.busy, els),
    showError: showToast,
    refresh,
  });
}

els.runtimeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const runtime = button.dataset.runtime;
    const request = runtimeRequest(state.snapshot, runtime);
    if (!request) return;
    perform(BusyKey.RUNTIME, () => post(request.path, request.payload));
  });
});

els.robotSelect.addEventListener("change", () => {
  const robot = els.robotSelect.value;
  const request = robotRequest(state.snapshot, robot);
  if (!request) return;
  perform(BusyKey.ROBOT, () => post(request.path, request.payload));
});

els.sceneSelect.addEventListener("change", () => {
  const scene = els.sceneSelect.value;
  perform(BusyKey.SCENE, () =>
    post(API.SCENE, {
      [ApiKey.SCENE]: scene,
      ...safetyPayload(state.snapshot),
    }),
  );
});

els.stackAction.addEventListener("click", () => {
  const request = processRequest(
    state.snapshot,
    ProcessName.STACK,
    window.confirm,
  );
  if (!request) return;
  perform(ProcessName.STACK, () => post(request.path, request.payload));
});

Object.values(els.processes).forEach(({ action: button }) => {
  button.addEventListener("click", () => {
    const name = button.dataset.processAction;
    const request = processRequest(state.snapshot, name, window.confirm);
    if (!request) return;
    perform(name, () => post(request.path, request.payload));
  });
});

els.recordingButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const request = recordingRequest(button.dataset.recordingAction);
    perform(BusyKey.RECORDINGS, () => post(request.path, request.payload));
  });
});

els.replayRecording.addEventListener("change", () => {
  render(state.snapshot, state.busy, els);
});

els.replayAction.addEventListener("click", () => {
  const request = replayRequest(
    state.snapshot,
    els.replayRecording.value,
    window.confirm,
  );
  if (!request) return;
  perform(BusyKey.REPLAY, () => post(request.path, request.payload));
});

els.orchestratorButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const payload = {
      [ApiKey.MODE]: button.dataset.orchestratorMode,
    };
    if (button.dataset.orchestratorPreset) {
      payload[ApiKey.PARAMETERS] = {
        [OrchestratorParameter.PRESET]: button.dataset.orchestratorPreset,
      };
    }
    perform(BusyKey.ORCHESTRATOR, () => post(API.ORCHESTRATOR, payload));
  });
});

refresh();
startCameraStreams(els.cameras);
window.setInterval(() => {
  void refresh();
}, 1000);
