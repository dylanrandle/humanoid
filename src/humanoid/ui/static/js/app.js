import { fetchApplicationLogs, fetchStatus, post } from "./api.js";
import {
  API,
  BusyKey,
  OrchestratorParameter,
  PayloadKey,
  ProcessName,
} from "./constants.js";
import { els } from "./dom.js";
import { errorMessage, performOperation } from "./operations.js";
import {
  render,
  renderApplicationLogs,
  renderApplicationLogsDisconnected,
  renderDisconnected,
} from "./render.js";
import {
  collectionRequest,
  collectionRunning,
  collectionSceneRequest,
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
  applicationLogs: [],
  applicationLogCursor: null,
  applicationLogsConnected: null,
  refreshPromise: null,
  sceneStackPid: undefined,
  sceneSelectionPending: true,
};

function refresh() {
  if (state.refreshPromise) return state.refreshPromise;
  state.refreshPromise = Promise.all([
    loadStatus(),
    loadApplicationLogs(),
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
    syncTaskScene();
  } catch (error) {
    const message = errorMessage(error);
    renderDisconnected(message, els);
    if (state.connected !== false) showToast(message);
    state.connected = false;
  }
}

async function loadApplicationLogs() {
  try {
    const snapshot = await fetchApplicationLogs(state.applicationLogCursor);
    if (snapshot.reset) state.applicationLogs = [];
    state.applicationLogs = [
      ...state.applicationLogs,
      ...snapshot.entries.map((entry) => entry.message),
    ].slice(-snapshot.capacity);
    state.applicationLogCursor = snapshot.cursor;
    state.applicationLogsConnected = true;
    renderApplicationLogs(state.applicationLogs, els);
  } catch (error) {
    renderApplicationLogsDisconnected(els);
    if (state.applicationLogsConnected !== false)
      showToast(errorMessage(error));
    state.applicationLogsConnected = false;
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

function syncTaskScene() {
  const snapshot = state.snapshot;
  const stack = snapshot?.processes[ProcessName.STACK];
  const pid = stack?.running ? stack.pid : null;
  if (pid !== state.sceneStackPid) {
    // Reconnecting to an existing task world must not reset its physics.
    state.sceneSelectionPending = !(
      state.sceneStackPid === undefined && snapshot?.collection?.task
    );
    state.sceneStackPid = pid;
  }
  if (!state.sceneSelectionPending || state.busy.size) return;
  const request = collectionSceneRequest(
    snapshot,
    els.collectionFields.task.value,
    els.collectionFields.seed.valueAsNumber,
  );
  if (!request) return;
  state.sceneSelectionPending = false;
  perform(BusyKey.COLLECTION, () => post(request.path, request.payload));
}

for (const name of ["task", "seed"]) {
  els.collectionFields[name].addEventListener("change", () => {
    state.sceneSelectionPending = true;
    render(state.snapshot, state.busy, els);
    syncTaskScene();
  });
}
els.collectionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const fields = els.collectionFields;
  const settings = Object.fromEntries(
    Object.entries(fields).map(([name, field]) => [
      name,
      field.type === "number" ? field.valueAsNumber : field.value,
    ]),
  );
  if (
    !collectionRunning(state.snapshot?.collection) &&
    settings.max_attempts < settings.episodes
  ) {
    showToast(
      "Maximum attempts must be at least the number of successful episodes.",
    );
    return;
  }
  const request = collectionRequest(state.snapshot, settings);
  if (request)
    perform(BusyKey.COLLECTION, () => post(request.path, request.payload));
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
