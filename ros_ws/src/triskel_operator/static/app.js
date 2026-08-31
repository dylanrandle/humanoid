const state = { snapshot: null, held: new Set(), busy: false, teleopBusy: false };

const elements = {
  systemPill: document.querySelector("#system-pill"),
  systemLabel: document.querySelector("#system-label"),
  robot: document.querySelector("#robot-select"),
  runtime: document.querySelector("#runtime-value"),
  mode: document.querySelector("#mode-value"),
  runtimeNote: document.querySelector("#runtime-note"),
  controllerDots: [...document.querySelectorAll("[data-controller]")],
  modeButtons: [...document.querySelectorAll("[data-mode]")],
  commandButtons: [...document.querySelectorAll("[data-command]")],
  jointCount: document.querySelector("#joint-count"),
  odomState: document.querySelector("#odom-state"),
  runtimeState: document.querySelector("#runtime-state"),
  feedbackAge: document.querySelector("#feedback-age"),
  topicRateGrid: document.querySelector("#topic-rate-grid"),
  visualizationFrame: document.querySelector("#visualization-frame"),
  visualizationStatus: document.querySelector("#visualization-status"),
  visualizationMessage: document.querySelector("#visualization-message"),
  visualizationOpen: document.querySelector("#visualization-open"),
  recordAction: document.querySelector("#record-action"),
  recordDetail: document.querySelector("#record-detail"),
  recordingSelect: document.querySelector("#recording-select"),
  replayAction: document.querySelector("#replay-action"),
  replayDetail: document.querySelector("#replay-detail"),
  error: document.querySelector("#error"),
  toast: document.querySelector("#toast"),
};

const keyCommands = new Map([
  ["KeyD", "tool_forward"], ["KeyA", "tool_backward"],
  ["KeyW", "tool_left"], ["KeyS", "tool_right"],
  ["KeyE", "tool_up"], ["KeyQ", "tool_down"],
  ["KeyK", "tool_roll_left"], ["KeyI", "tool_roll_right"],
  ["KeyO", "tool_pitch_up"], ["KeyU", "tool_pitch_down"],
  ["KeyJ", "tool_yaw_left"], ["KeyL", "tool_yaw_right"],
  ["ArrowUp", "base_forward"], ["ArrowDown", "base_backward"],
  ["ArrowLeft", "base_left"], ["ArrowRight", "base_right"],
  ["Comma", "base_yaw_left"], ["Period", "base_yaw_right"],
  ["BracketLeft", "gripper_close"], ["BracketRight", "gripper_open"],
]);

async function request(path, payload = null) {
  const options = payload === null ? { cache: "no-store" } : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
  const response = await fetch(path, options);
  const result = await response.json().catch(() => null);
  if (!response.ok || !result?.ok) throw new Error(result?.error || `Request failed (${response.status})`);
  return result.status;
}

function hardwareAcknowledged(action) {
  if (state.snapshot?.runtime !== "real") return true;
  return window.confirm(`${action} can move the physical Triskel robot. Confirm the workspace is clear.`);
}

async function operation(action) {
  if (state.busy) return;
  state.busy = true;
  try {
    state.snapshot = await action();
    render();
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    state.busy = false;
    await refresh();
  }
}

async function refresh() {
  try {
    state.snapshot = await request("/api/status");
    render();
  } catch (error) {
    elements.systemPill.classList.remove("online");
    elements.systemLabel.textContent = "Dashboard disconnected";
    showError(error instanceof Error ? error.message : String(error));
  }
}

function render() {
  const snapshot = state.snapshot;
  if (!snapshot) return;
  if (snapshot.mode !== "keyboard") state.held.clear();
  elements.systemPill.classList.toggle("online", snapshot.ready);
  elements.systemLabel.textContent = snapshot.ready ? "ROS graph online" : "ROS graph starting";
  elements.robot.disabled = false;
  elements.robot.value = snapshot.robot;
  elements.runtime.textContent = snapshot.runtime === "sim" ? "Simulation" : "Real hardware";
  elements.mode.textContent = snapshot.preset ? `${snapshot.preset} · moving` : snapshot.mode;
  renderVisualization(snapshot.visualization);
  elements.runtimeNote.textContent = snapshot.runtime === "sim"
    ? "Mock ros2_control hardware is active; no physical device is opened."
    : "Physical Feetech hardware is active; motion requires confirmation.";

  for (const dot of elements.controllerDots) {
    const name = dot.dataset.controller;
    const online = name === "servo"
      ? snapshot.servo_ready
      : name === "meta_quest"
        ? snapshot.meta_quest.ready
        : name === "visualization" ? snapshot.visualization.ready : snapshot.controllers[name];
    dot.classList.toggle("online", online);
    dot.parentElement.querySelector("b").textContent = online ? "Online" : "Waiting";
  }
  for (const button of elements.modeButtons) {
    const mode = button.dataset.mode;
    const active = mode === snapshot.mode || (snapshot.mode === "homing" && mode === snapshot.preset);
    button.classList.toggle("active", active);
    button.disabled = state.busy || (!snapshot.ready && mode !== "idle") || snapshot.replay.running;
  }
  for (const button of elements.commandButtons) {
    button.disabled = snapshot.mode !== "keyboard";
    button.classList.toggle("active", state.held.has(button.dataset.command));
  }

  const jointCount = Object.keys(snapshot.joints).filter((name) => name.startsWith("arm_") || name === "gripper_1").length;
  elements.jointCount.textContent = `${jointCount} / 8`;
  elements.odomState.textContent = ageLabel(snapshot.odometry_age);
  elements.runtimeState.textContent = snapshot.runtime === "sim" ? "Mock hardware" : "Feetech bus";
  elements.feedbackAge.textContent = snapshot.joint_state_age === null ? "No samples" : `${snapshot.joint_state_age.toFixed(2)}s old`;
  renderTopicRates(snapshot.topic_rates);

  elements.recordAction.textContent = snapshot.recording.running ? "Stop recording" : "Start recording";
  elements.recordAction.classList.toggle("stop", snapshot.recording.running);
  elements.recordAction.disabled = state.busy || !snapshot.ready || snapshot.replay.running;
  elements.recordDetail.textContent = snapshot.recording.running ? `Recording ${snapshot.recording.id}` : "Stopped";
  renderRecordings(snapshot.recordings);
  elements.replayAction.textContent = snapshot.replay.running ? "Stop replay" : "Start replay";
  elements.replayAction.classList.toggle("stop", snapshot.replay.running);
  elements.replayAction.disabled = state.busy || !snapshot.ready || snapshot.recording.running || (!snapshot.replay.running && !elements.recordingSelect.value);
  elements.recordingSelect.disabled = snapshot.replay.running || snapshot.recording.running;
  elements.replayDetail.textContent = snapshot.replay.running
    ? `Playing ${snapshot.replay.id}`
    : snapshot.replay.outcome || "Ready";
  elements.error.hidden = !snapshot.last_error;
  elements.error.textContent = snapshot.last_error || "";
}

function renderTopicRates(rates) {
  if (!Array.isArray(rates) || rates.length === 0) {
    const empty = document.createElement("p");
    empty.className = "topic-rate-empty";
    empty.textContent = "Waiting for ROS samples.";
    elements.topicRateGrid.replaceChildren(empty);
    return;
  }
  const cards = rates.map((rate) => {
    const card = document.createElement("article");
    card.className = `topic-rate-card ${rate.state}`;

    const heading = document.createElement("div");
    const dot = document.createElement("span");
    dot.className = "topic-rate-dot";
    const label = document.createElement("b");
    label.textContent = rate.label;
    heading.append(dot, label);

    const frequency = document.createElement("strong");
    frequency.textContent = rate.state === "idle" ? "Idle" : `${Number(rate.hz).toFixed(1)} Hz`;
    const topic = document.createElement("code");
    topic.textContent = rate.topic;
    const detail = document.createElement("small");
    detail.textContent = rate.state === "idle"
      ? "Monitored when active"
      : `Minimum ${Number(rate.minimum_hz).toFixed(1)} Hz`;
    card.append(heading, frequency, topic, detail);
    return card;
  });
  elements.topicRateGrid.replaceChildren(...cards);
}

function renderVisualization(visualization) {
  const enabled = Boolean(visualization?.enabled);
  const ready = enabled && Boolean(visualization?.ready);
  elements.visualizationStatus.textContent = enabled ? (ready ? "Live ROS view" : "Starting Viser") : "Disabled";
  elements.visualizationMessage.hidden = ready;
  elements.visualizationOpen.hidden = !enabled;
  if (!enabled) {
    elements.visualizationFrame.removeAttribute("src");
    elements.visualizationFrame.dataset.ready = "false";
    return;
  }

  const url = `http://${window.location.hostname}:${visualization.port}/`;
  const becameReady = ready && elements.visualizationFrame.dataset.ready !== "true";
  if (elements.visualizationFrame.dataset.url !== url || becameReady) {
    elements.visualizationFrame.src = url;
    elements.visualizationFrame.dataset.url = url;
  }
  elements.visualizationFrame.dataset.ready = String(ready);
  elements.visualizationOpen.href = url;
}

function renderRecordings(recordings) {
  const current = elements.recordingSelect.value;
  const values = recordings.map((recording) => recording.id);
  const existing = [...elements.recordingSelect.options].slice(1).map((option) => option.value);
  if (values.join("|") !== existing.join("|")) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = recordings.length ? "Select recording" : "No recordings";
    const options = recordings.map((recording) => {
      const option = document.createElement("option");
      option.value = recording.id;
      option.textContent = `${recording.id} · ${recording.runtime}`;
      return option;
    });
    elements.recordingSelect.replaceChildren(placeholder, ...options);
    if (values.includes(current)) elements.recordingSelect.value = current;
  }
}

function ageLabel(age) {
  if (age === null) return "Waiting";
  return age < 2 ? "Streaming" : "Stale";
}

function showError(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showError.timer);
  showError.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 5000);
}

elements.robot.addEventListener("change", () => operation(() => request("/api/robot", { robot: elements.robot.value })));
elements.modeButtons.forEach((button) => button.addEventListener("click", () => {
  const mode = button.dataset.mode;
  if (mode !== "idle" && !hardwareAcknowledged(`Entering ${mode} mode`)) return;
  operation(() => request("/api/mode", { mode, acknowledge_real_hardware: state.snapshot.runtime === "real" }));
}));
elements.recordAction.addEventListener("click", () => operation(() => request(
  state.snapshot.recording.running ? "/api/recording/stop" : "/api/recording/start", {}
)));
elements.replayAction.addEventListener("click", () => {
  if (state.snapshot.replay.running) {
    operation(() => request("/api/replay/stop", {}));
    return;
  }
  if (!hardwareAcknowledged("Replay")) return;
  operation(() => request("/api/replay/start", {
    recording: elements.recordingSelect.value,
    acknowledge_real_hardware: state.snapshot.runtime === "real",
  }));
});
elements.recordingSelect.addEventListener("change", render);

function hold(command, active) {
  if (!command) return;
  if (!active) {
    state.held.delete(command);
    render();
    if (state.snapshot?.mode === "keyboard") void sendTeleop();
    return;
  }
  if (state.snapshot?.mode !== "keyboard") return;
  state.held.add(command);
  render();
  void sendTeleop();
}

elements.commandButtons.forEach((button) => {
  const command = button.dataset.command;
  button.addEventListener("pointerdown", (event) => { event.preventDefault(); button.setPointerCapture(event.pointerId); hold(command, true); });
  for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
    button.addEventListener(eventName, () => hold(command, false));
  }
});
window.addEventListener("keydown", (event) => {
  const command = keyCommands.get(event.code);
  if (command) { event.preventDefault(); hold(command, true); }
});
window.addEventListener("keyup", (event) => {
  const command = keyCommands.get(event.code);
  if (command) { event.preventDefault(); hold(command, false); }
});
window.addEventListener("blur", () => { state.held.clear(); void sendTeleop(); render(); });

async function sendTeleop() {
  if (state.teleopBusy || state.snapshot?.mode !== "keyboard") return;
  state.teleopBusy = true;
  try {
    state.snapshot = await request("/api/teleop", { commands: [...state.held] });
  } catch (error) {
    state.held.clear();
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    state.teleopBusy = false;
  }
}

window.setInterval(() => { if (state.snapshot?.mode === "keyboard") void sendTeleop(); }, 100);
window.setInterval(() => { void refresh(); }, 1000);
void refresh();
