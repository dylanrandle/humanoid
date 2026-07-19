import { fetchStatus, post } from "./api.js";
import {
  API,
  BusyKey,
  OrchestratorParameter,
  PayloadKey,
  ProcessName,
} from "./constants.js";
import { els } from "./dom.js";
import { errorMessage, performOperation } from "./operations.js";
import { render, renderDisconnected } from "./render.js";
import {
  loggingRequest,
  processRequest,
  replayRequest,
  robotRequest,
  runtimeRequest,
} from "./requests.js";
import { showToast } from "./toast.js";

const state = {
  snapshot: null,
  busy: new Set(),
  connected: null,
  refreshPromise: null,
};

function refresh() {
  if (state.refreshPromise) return state.refreshPromise;
  state.refreshPromise = loadStatus().finally(() => {
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
    const request = runtimeRequest(state.snapshot, runtime, window.confirm);
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

els.loggingButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const request = loggingRequest(button.dataset.loggingAction);
    perform(BusyKey.LOGGING, () => post(request.path, request.payload));
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
      [PayloadKey.MODE]: button.dataset.orchestratorMode,
    };
    if (button.dataset.orchestratorPreset) {
      payload[PayloadKey.PARAMETERS] = {
        [OrchestratorParameter.PRESET]: button.dataset.orchestratorPreset,
      };
    }
    perform(BusyKey.ORCHESTRATOR, () => post(API.ORCHESTRATOR, payload));
  });
});

refresh();
window.setInterval(() => {
  void refresh();
}, 1000);
