import {
  BusyKey,
  LoggingState,
  Mode,
  OrchestratorParameter,
  PayloadKey,
  ProcessName,
  ReplayOutcome,
  Runtime,
  TELEOP_PROCESSES,
} from "./constants.js";
export function render(snapshot, busy, elements) {
  if (!snapshot) return;

  const processes = snapshot.processes;
  const stack = processes[ProcessName.STACK];
  const stackActive = stack.running;
  const replayActive =
    snapshot.replay.running || processes[ProcessName.REPLAY].running;
  const controlsReady = stackActive && snapshot.orchestrator.connected;
  const anyRunning = Object.values(processes).some(
    (process) => process.running,
  );
  const configurationBusy =
    busy.has(BusyKey.RUNTIME) ||
    busy.has(BusyKey.ROBOT) ||
    busy.has(BusyKey.SCENE);

  elements.robotName.textContent = formatRobotName(snapshot.robot);
  elements.systemState.classList.toggle(
    "online",
    controlsReady || replayActive,
  );
  elements.systemStateLabel.textContent = systemStateLabel(
    stack.running,
    replayActive,
    controlsReady,
  );

  for (const button of elements.runtimeButtons) {
    const selected = button.dataset.runtime === snapshot.runtime;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.disabled = anyRunning || configurationBusy;
  }
  renderRobotSelector(elements.robotSelect, snapshot.robots, snapshot.robot);
  elements.robotSelect.disabled = anyRunning || configurationBusy;
  renderSceneSelector(elements.sceneSelect, snapshot.scenes, snapshot.scene);
  elements.sceneSelect.disabled =
    snapshot.runtime !== Runtime.SIM || anyRunning || configurationBusy;
  elements.configurationLockNote.textContent = anyRunning
    ? "Locked while active"
    : "Unlocked";
  elements.runtimeFootnote.textContent =
    snapshot.runtime === Runtime.SIM
      ? "Simulation does not command physical hardware."
      : "Real runtime can command physical servos. Clear the workspace before starting.";

  renderStack(
    stack,
    controlsReady,
    replayActive,
    snapshot.runtime,
    busy,
    elements,
  );
  renderNodeRates(snapshot.node_rates || [], elements);
  for (const name of TELEOP_PROCESSES) {
    renderProcess(name, processes[name], controlsReady, busy, elements);
  }
  renderLogging(snapshot.logging, controlsReady, busy, elements);
  renderReplay(snapshot, anyRunning, replayActive, busy, elements);
  renderOrchestrator(
    snapshot.orchestrator,
    processes,
    stackActive,
    controlsReady,
    busy,
    elements,
  );
}

function renderNodeRates(nodeRates, elements) {
  const rows = nodeRates.map((rate) => {
    const row = elements.nodeRateList.ownerDocument.createElement("article");
    row.className = "node-rate-row";
    row.classList.toggle("healthy", rate.healthy);
    row.classList.toggle("unhealthy", !rate.healthy);
    row.setAttribute("data-node-rate", rate.node_name);

    const dot = elements.nodeRateList.ownerDocument.createElement("span");
    dot.className = "node-rate-dot";
    dot.setAttribute("aria-hidden", "true");

    const copy = elements.nodeRateList.ownerDocument.createElement("div");
    copy.className = "node-rate-copy";
    const name = elements.nodeRateList.ownerDocument.createElement("strong");
    name.textContent = formatNodeName(rate.node_name);
    const detail = elements.nodeRateList.ownerDocument.createElement("small");
    detail.textContent = rateDetail(rate);
    copy.replaceChildren(name, detail);

    const state = elements.nodeRateList.ownerDocument.createElement("span");
    state.className = "node-rate-state";
    state.textContent = rate.healthy ? "Healthy" : "Unhealthy";
    row.replaceChildren(dot, copy, state);
    return row;
  });

  if (rows.length === 0) {
    const empty = elements.nodeRateList.ownerDocument.createElement("p");
    empty.className = "node-rate-empty";
    empty.textContent = "Start a process to view its loop rate.";
    elements.nodeRateList.replaceChildren(empty);
    elements.nodeRateSummary.textContent = "No active nodes";
    elements.nodeRateSummary.classList.remove("healthy", "unhealthy");
    return;
  }

  elements.nodeRateList.replaceChildren(...rows);
  const healthyCount = nodeRates.filter((rate) => rate.healthy).length;
  const allHealthy = healthyCount === nodeRates.length;
  elements.nodeRateSummary.textContent = `${healthyCount} of ${nodeRates.length} healthy`;
  elements.nodeRateSummary.classList.toggle("healthy", allHealthy);
  elements.nodeRateSummary.classList.toggle("unhealthy", !allHealthy);
}

function rateDetail(rate) {
  if (rate.target_rate_hz === null) return "Waiting for telemetry";
  if (rate.measured_rate_hz === null) {
    return `Measuring · target ${formatRate(rate.target_rate_hz)}`;
  }
  return `${formatRate(rate.measured_rate_hz)} measured · ${formatRate(rate.target_rate_hz)} target`;
}

function formatRate(rate) {
  return `${rate.toFixed(1)} Hz`;
}

function formatNodeName(name) {
  return name.replace(/Node$/, "").replace(/([a-z0-9])([A-Z])/g, "$1 $2");
}

function renderLogging(logging, controlsReady, busy, elements) {
  const isBusy = busy.has(BusyKey.LOGGING);
  const active =
    logging.state === LoggingState.STARTING ||
    logging.state === LoggingState.RUNNING;

  for (const button of elements.loggingButtons) {
    const startsLogging = button.dataset.loggingAction === "start";
    button.disabled =
      !controlsReady ||
      isBusy ||
      (startsLogging
        ? active || logging.state === LoggingState.STOPPING
        : !active);
  }

  const labels = {
    [LoggingState.STOPPED]: "Stopped",
    [LoggingState.STARTING]: "Starting",
    [LoggingState.RUNNING]: "Recording",
    [LoggingState.STOPPING]: "Stopping",
    [LoggingState.FAILED]: "Failed",
  };
  elements.loggingStatus.textContent = labels[logging.state] || "Unknown";

  if (logging.state === LoggingState.RUNNING) {
    elements.loggingDetail.textContent = logging.file_name
      ? `Recording to ${logging.file_name}.`
      : "Recording LCM traffic.";
  } else if (logging.state === LoggingState.STARTING) {
    elements.loggingDetail.textContent = "Starting lcm-logger…";
  } else if (logging.state === LoggingState.STOPPING) {
    elements.loggingDetail.textContent =
      "Stopping and flushing the current log…";
  } else if (logging.state === LoggingState.FAILED) {
    elements.loggingDetail.textContent = "Data logging is unavailable.";
  } else {
    elements.loggingDetail.textContent =
      "Logs are written to the project logs directory.";
  }

  const failed = logging.state === LoggingState.FAILED;
  elements.loggingError.hidden = !failed;
  elements.loggingError.textContent = failed
    ? logging.error || "Data logging failed."
    : "";
}

function renderStack(
  stack,
  controlsReady,
  replayActive,
  runtime,
  busy,
  elements,
) {
  const running = stack.running;
  elements.stackLight.classList.toggle("online", running);
  elements.stackAction.classList.toggle("stop", running);
  elements.stackAction.classList.toggle(
    "real-start",
    !running && runtime === Runtime.REAL,
  );
  elements.stackAction.disabled =
    busy.has(ProcessName.STACK) || (!running && replayActive);

  if (busy.has(ProcessName.STACK)) {
    elements.stackStatus.textContent = running ? "Stopping" : "Starting";
    elements.stackDetail.textContent = "Waiting for process";
    elements.stackActionLabel.textContent = "Working…";
  } else if (running && !controlsReady) {
    elements.stackStatus.textContent = "Starting";
    elements.stackDetail.textContent =
      "Waiting for robot state and orchestrator";
    elements.stackActionLabel.textContent = "Stop stack";
  } else if (running) {
    elements.stackStatus.textContent = "Running";
    elements.stackDetail.textContent = `PID ${stack.pid} · ${formatUptime(stack.uptime_seconds)}`;
    elements.stackActionLabel.textContent = "Stop stack";
  } else {
    elements.stackStatus.textContent =
      stack.exit_code && stack.exit_code !== 0
        ? `Exited (${stack.exit_code})`
        : "Stopped";
    elements.stackDetail.textContent =
      stack.last_output ||
      `Ready in ${runtime === Runtime.SIM ? "simulation" : "real"} mode`;
    elements.stackActionLabel.textContent =
      runtime === Runtime.REAL ? "Start real stack" : "Start stack";
  }
}

function renderReplay(snapshot, anyRunning, replayActive, busy, elements) {
  const replay = snapshot.replay;
  const isBusy = busy.has(BusyKey.REPLAY);
  renderRecordingSelector(elements.replayRecording, snapshot.recordings);
  const selectedRecording =
    snapshot.recordings.find(
      (recording) => recording.id === elements.replayRecording.value,
    ) || null;
  const compatible = selectedRecording?.robot === snapshot.robot;

  elements.replayRecording.disabled =
    replayActive || isBusy || snapshot.recordings.length === 0;
  elements.replayAction.textContent = isBusy
    ? replayActive
      ? "Stopping…"
      : "Starting…"
    : replayActive
      ? "Stop replay"
      : "Play replay";
  elements.replayAction.disabled =
    isBusy ||
    (!replayActive && (!selectedRecording || !compatible || anyRunning));

  if (replayActive) {
    elements.replayStatus.textContent = "Playing";
    elements.replayDetail.textContent =
      "Publishing recorded command and mode channels.";
  } else if (replay.outcome === ReplayOutcome.COMPLETED) {
    elements.replayStatus.textContent = "Complete";
    elements.replayDetail.textContent =
      "Replay finished. Select Play replay to run it again.";
  } else if (replay.outcome === ReplayOutcome.STOPPED) {
    elements.replayStatus.textContent = "Stopped";
    elements.replayDetail.textContent = "Replay was stopped by the operator.";
  } else if (replay.outcome === ReplayOutcome.FAILED) {
    elements.replayStatus.textContent = "Failed";
    elements.replayDetail.textContent = "Replay stopped before completing.";
  } else if (anyRunning) {
    elements.replayStatus.textContent = "Unavailable";
    elements.replayDetail.textContent =
      "Stop the stack and teleop nodes before replaying.";
  } else if (snapshot.recordings.length === 0) {
    elements.replayStatus.textContent = "Unavailable";
    elements.replayDetail.textContent =
      "Create a recording before starting replay.";
  } else if (selectedRecording && !compatible) {
    elements.replayStatus.textContent = "Incompatible";
    elements.replayDetail.textContent = `Select the ${formatRobotName(selectedRecording.robot)} robot.`;
  } else if (snapshot.runtime === Runtime.REAL) {
    elements.replayStatus.textContent = "Hardware";
    elements.replayDetail.textContent =
      "Replay will command real hardware after confirmation.";
  } else {
    elements.replayStatus.textContent = "Ready";
    elements.replayDetail.textContent = selectedRecording
      ? "Ready to replay through the simulator."
      : "Select a recording from the project logs directory.";
  }

  const failed = !replayActive && replay.outcome === ReplayOutcome.FAILED;
  elements.replayError.hidden = !failed;
  elements.replayError.textContent = failed
    ? replay.last_output || "Replay failed."
    : "";
}

function renderProcess(name, process, controlsReady, busy, rootElements) {
  const elements = rootElements.processes[name];
  const isBusy = busy.has(name);

  elements.row.classList.toggle("running", process.running);
  elements.status.textContent = isBusy
    ? process.running
      ? "Stopping"
      : "Starting"
    : process.running
      ? "Running"
      : "Stopped";
  elements.action.textContent = isBusy
    ? "Working…"
    : process.running
      ? "Stop"
      : "Start";
  elements.action.disabled = isBusy || (!process.running && !controlsReady);

  const failed =
    !process.running && process.exit_code !== null && process.exit_code !== 0;
  elements.error.hidden = !failed;
  elements.error.textContent = failed
    ? process.last_output || `Exited with code ${process.exit_code}`
    : "";
}

function renderOrchestrator(
  orchestrator,
  processes,
  stackActive,
  controlsReady,
  busy,
  elements,
) {
  const mode = orchestrator[PayloadKey.MODE];
  const preset =
    orchestrator[PayloadKey.PARAMETERS][OrchestratorParameter.PRESET] || null;
  elements.activeMode.textContent = orchestratorLabel(
    mode,
    preset,
    stackActive,
  );
  elements.modePulse.classList.toggle("online", Boolean(mode));

  for (const button of elements.orchestratorButtons) {
    const requestedMode = button.dataset.orchestratorMode;
    const requestedPreset = button.dataset.orchestratorPreset || null;
    const active = requestedPreset
      ? requestedMode === Mode.HOMING && requestedPreset === preset
      : preset === null && requestedMode === mode;
    const teleopUnavailable =
      TELEOP_PROCESSES.includes(requestedMode) &&
      !processes[requestedMode].running;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled =
      !controlsReady || busy.has(BusyKey.ORCHESTRATOR) || teleopUnavailable;
  }
}

export function renderDisconnected(
  message = "Check the start process",
  elements,
) {
  elements.systemState.classList.remove("online");
  elements.systemStateLabel.textContent = "Console disconnected";
  elements.stackLight.classList.remove("online");
  elements.stackStatus.textContent = "Unavailable";
  elements.stackDetail.textContent = message;
  elements.stackAction.disabled = true;
  elements.nodeRateSummary.textContent = "Unavailable";
  elements.nodeRateSummary.classList.remove("healthy", "unhealthy");
  const nodeRateMessage =
    elements.nodeRateList.ownerDocument.createElement("p");
  nodeRateMessage.className = "node-rate-empty";
  nodeRateMessage.textContent = "Reconnect to view node-rate health.";
  elements.nodeRateList.replaceChildren(nodeRateMessage);
  elements.runtimeButtons.forEach((button) => {
    button.disabled = true;
  });
  elements.robotSelect.disabled = true;
  elements.sceneSelect.disabled = true;
  Object.values(elements.processes).forEach(({ action }) => {
    action.disabled = true;
  });
  elements.loggingButtons.forEach((button) => {
    button.disabled = true;
  });
  elements.loggingStatus.textContent = "Unavailable";
  elements.loggingDetail.textContent = "Reconnect to view data logging status.";
  elements.loggingError.hidden = true;
  elements.replayRecording.disabled = true;
  elements.replayAction.disabled = true;
  elements.replayStatus.textContent = "Unavailable";
  elements.replayDetail.textContent = "Reconnect to view replay status.";
  elements.replayError.hidden = true;
  elements.orchestratorButtons.forEach((button) => {
    button.disabled = true;
  });
}

function renderRecordingSelector(select, recordings) {
  const current = select.value;
  const expectedValues = ["", ...recordings.map((recording) => recording.id)];
  const currentValues = [...select.options].map((option) => option.value);
  if (
    currentValues.length !== expectedValues.length ||
    currentValues.some((value, index) => value !== expectedValues[index])
  ) {
    const placeholder = select.ownerDocument.createElement("option");
    placeholder.value = "";
    placeholder.textContent = recordings.length
      ? "Select a recording"
      : "No recordings available";
    const options = recordings.map((recording) => {
      const option = select.ownerDocument.createElement("option");
      option.value = recording.id;
      option.textContent = `${recording.id} · ${formatRobotName(recording.robot)}`;
      return option;
    });
    select.replaceChildren(placeholder, ...options);
  }
  select.value = recordings.some((recording) => recording.id === current)
    ? current
    : "";
}

function renderRobotSelector(select, robots, selectedRobot) {
  const currentRobots = [...select.options].map((option) => option.value);
  if (
    currentRobots.length !== robots.length ||
    currentRobots.some((robot, index) => robot !== robots[index])
  ) {
    const options = robots.map((robot) => {
      const option = select.ownerDocument.createElement("option");
      option.value = robot;
      option.textContent = formatRobotName(robot);
      return option;
    });
    select.replaceChildren(...options);
  }
  select.value = selectedRobot;
}

function renderSceneSelector(select, scenes, selectedScene) {
  const currentScenes = [...select.options].map((option) => option.value);
  if (
    currentScenes.length !== scenes.length ||
    currentScenes.some((scene, index) => scene !== scenes[index])
  ) {
    const options = scenes.map((scene) => {
      const option = select.ownerDocument.createElement("option");
      option.value = scene;
      option.textContent = scene;
      return option;
    });
    select.replaceChildren(...options);
  }
  select.value = selectedScene;
}

function formatRobotName(name) {
  return name.replaceAll("_", " ");
}

function systemStateLabel(stackRunning, replayRunning, controlsReady) {
  if (controlsReady) return "System online";
  if (replayRunning) return "Replay active";
  return stackRunning ? "Stack starting" : "System offline";
}

function orchestratorLabel(mode, preset, stackActive) {
  if (preset) {
    return `${preset} · ${mode === Mode.HOMING ? "moving" : "requested"}`;
  }
  if (mode) return mode.replaceAll("_", " ");
  return stackActive ? "Connecting" : "Offline";
}

function formatUptime(seconds) {
  if (seconds === null) return "starting";
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}m ${remainder}s`;
}
