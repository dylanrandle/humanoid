import assert from "node:assert/strict";
import test from "node:test";

import { ProcessName } from "../../../src/humanoid/ui/static/js/constants.js";
import { render, renderDisconnected } from "../../../src/humanoid/ui/static/js/render.js";

class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  toggle(name, enabled) {
    if (enabled) this.values.add(name);
    else this.values.delete(name);
  }

  remove(name) {
    this.values.delete(name);
  }
}

function element(dataset = {}) {
  return {
    classList: new FakeClassList(),
    dataset,
    disabled: false,
    hidden: true,
    textContent: "",
    value: "",
    options: [],
    children: [],
    ownerDocument: { createElement: () => element() },
    setAttribute(name, value) { this[name] = value; },
    replaceChildren(...children) {
      this.options = children;
      this.children = children;
    },
  };
}

function elements() {
  return {
    systemState: element(),
    systemStateLabel: element(),
    robotName: element(),
    robotSelect: element(),
    sceneSelect: element(),
    runtimeButtons: [element({ runtime: "sim" }), element({ runtime: "real" })],
    configurationLockNote: element(),
    runtimeFootnote: element(),
    stackLight: element(),
    stackStatus: element(),
    stackDetail: element(),
    stackAction: element(),
    stackActionLabel: element(),
    nodeRateSummary: element(),
    nodeRateList: element(),
    processes: {
      keyboard: { row: element(), status: element(), error: element(), action: element() },
      oculus: { row: element(), status: element(), error: element(), action: element() },
    },
    loggingButtons: [element({ loggingAction: "start" }), element({ loggingAction: "stop" })],
    loggingStatus: element(),
    loggingDetail: element(),
    loggingError: element(),
    replayRecording: element(),
    replayStatus: element(),
    replayDetail: element(),
    replayError: element(),
    replayAction: element(),
    orchestratorButtons: [element({ orchestratorMode: "idle" })],
    activeMode: element(),
    modePulse: element(),
  };
}

test("node target and measured rates render healthy and unhealthy states", () => {
  const ui = elements();
  const current = snapshot();
  current.node_rates = [
    {
      node_name: "RobotControllerNode",
      pid: 101,
      target_rate_hz: 500,
      measured_rate_hz: 492.5,
      healthy: true,
      age_seconds: 0.1,
    },
    {
      node_name: "RobotVisualizerNode",
      pid: 102,
      target_rate_hz: 30,
      measured_rate_hz: 22,
      healthy: false,
      age_seconds: 0.2,
    },
  ];

  render(current, new Set(), ui);

  assert.equal(ui.nodeRateSummary.textContent, "1 of 2 healthy");
  assert.ok(ui.nodeRateSummary.classList.values.has("unhealthy"));
  assert.equal(ui.nodeRateList.children.length, 2);
  assert.ok(ui.nodeRateList.children[0].classList.values.has("healthy"));
  assert.ok(ui.nodeRateList.children[1].classList.values.has("unhealthy"));
  assert.equal(
    ui.nodeRateList.children[0].children[1].children[1].textContent,
    "492.5 Hz measured · 500.0 Hz target",
  );
});

function processStatus(overrides = {}) {
  return {
    running: false,
    pid: null,
    exit_code: null,
    runtime: null,
    uptime_seconds: null,
    last_output: null,
    ...overrides,
  };
}

function snapshot() {
  return {
    runtime: "sim",
    robot: "panda",
    robots: ["panda"],
    scene: "empty",
    scenes: ["empty", "floor-and-cube"],
    processes: {
      stack: processStatus(),
      replay: processStatus(),
      keyboard: processStatus(),
      oculus: processStatus(),
    },
    node_rates: [],
    logging: {
      timestamp: 0,
      state: "stopped",
      file_name: null,
      error: null,
    },
    recordings: [],
    replay: {
      running: false,
      file_name: null,
      outcome: null,
      exit_code: null,
      last_output: null,
    },
    orchestrator: {
      mode: null,
      connected: false,
      age_seconds: null,
      parameters: {},
    },
  };
}

test("logging lifecycle controls and failures are rendered", () => {
  const ui = elements();
  const current = snapshot();
  current.processes.stack = processStatus({ running: true });
  current.orchestrator = {
    mode: "idle",
    connected: true,
    age_seconds: 0,
    parameters: {},
  };

  current.logging = {
    timestamp: 1,
    state: "running",
    file_name: "logs/lcmlog_20260101",
    error: null,
  };
  render(current, new Set(), ui);
  assert.equal(ui.loggingStatus.textContent, "Recording");
  assert.equal(ui.loggingButtons[0].disabled, true);
  assert.equal(ui.loggingButtons[1].disabled, false);
  assert.match(ui.loggingDetail.textContent, /logs\/lcmlog_20260101/);

  current.logging = {
    timestamp: 2,
    state: "failed",
    file_name: null,
    error: "lcm-logger was not found in PATH.",
  };
  render(current, new Set(), ui);
  assert.equal(ui.loggingStatus.textContent, "Failed");
  assert.equal(ui.loggingError.hidden, false);
  assert.equal(ui.loggingError.textContent, "lcm-logger was not found in PATH.");
});

test("process failures and busy state are rendered", () => {
  const ui = elements();
  const current = snapshot();
  current.processes.keyboard = processStatus({
    exit_code: 7,
    last_output: "KeyboardTeleopNode exited",
  });

  render(current, new Set(), ui);
  assert.equal(ui.processes.keyboard.error.hidden, false);
  assert.equal(ui.processes.keyboard.error.textContent, "KeyboardTeleopNode exited");

  render(current, new Set([ProcessName.KEYBOARD]), ui);
  assert.equal(ui.processes.keyboard.status.textContent, "Starting");
  assert.equal(ui.processes.keyboard.action.textContent, "Working…");
  assert.equal(ui.processes.keyboard.action.disabled, true);
});

test("disconnected and external-stack errors disable every control", () => {
  const ui = elements();
  const message = "Another stack is already broadcasting. Stop it before using this console.";

  renderDisconnected(message, ui);

  assert.equal(ui.systemStateLabel.textContent, "Console disconnected");
  assert.equal(ui.stackDetail.textContent, message);
  assert.equal(ui.stackAction.disabled, true);
  assert.equal(ui.nodeRateSummary.textContent, "Unavailable");
  assert.match(ui.nodeRateList.children[0].textContent, /Reconnect/);
  assert.ok(ui.runtimeButtons.every((button) => button.disabled));
  assert.equal(ui.sceneSelect.disabled, true);
  assert.ok(Object.values(ui.processes).every(({ action }) => action.disabled));
  assert.ok(ui.loggingButtons.every((button) => button.disabled));
  assert.equal(ui.loggingStatus.textContent, "Unavailable");
  assert.equal(ui.loggingError.hidden, true);
  assert.equal(ui.replayRecording.disabled, true);
  assert.equal(ui.replayAction.disabled, true);
  assert.equal(ui.replayStatus.textContent, "Unavailable");
  assert.equal(ui.replayError.hidden, true);
  assert.ok(ui.orchestratorButtons.every((button) => button.disabled));
});

test("scene selection renders named scenes and locks outside stopped simulation", () => {
  const ui = elements();
  const current = snapshot();

  render(current, new Set(), ui);
  assert.deepEqual(
    ui.sceneSelect.options.map((option) => option.value),
    ["empty", "floor-and-cube"],
  );
  assert.equal(ui.sceneSelect.value, "empty");
  assert.equal(ui.sceneSelect.disabled, false);

  current.scene = "floor-and-cube";
  current.processes.stack = processStatus({ running: true });
  render(current, new Set(), ui);
  assert.equal(ui.sceneSelect.value, "floor-and-cube");
  assert.equal(ui.sceneSelect.disabled, true);

  current.processes.stack = processStatus();
  current.runtime = "real";
  render(current, new Set(), ui);
  assert.equal(ui.sceneSelect.disabled, true);
});

test("replay requires a compatible server-managed recording", () => {
  const ui = elements();
  const current = snapshot();

  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Unavailable");
  assert.equal(ui.replayAction.disabled, true);

  current.recordings = [{
    id: "recording_1",
    robot: "panda",
    created_at: "2026-01-01T12:00:00+00:00",
  }];
  render(current, new Set(), ui);
  assert.equal(ui.replayAction.disabled, true);
  ui.replayRecording.value = "recording_1";
  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Ready");
  assert.equal(ui.replayAction.disabled, false);

  current.runtime = "real";
  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Hardware");
  assert.equal(ui.replayAction.disabled, false);

  current.recordings[0].robot = "so101";
  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Incompatible");
  assert.equal(ui.replayAction.disabled, true);
});

test("active and failed replay states are rendered", () => {
  const ui = elements();
  const current = snapshot();
  current.processes.replay = processStatus({ running: true });
  current.replay = {
    running: true,
    file_name: "capture.lcmlog",
    outcome: null,
    exit_code: null,
    last_output: null,
  };

  render(current, new Set(), ui);
  assert.equal(ui.systemStateLabel.textContent, "Replay active");
  assert.equal(ui.replayStatus.textContent, "Playing");
  assert.equal(ui.replayAction.textContent, "Stop replay");
  assert.equal(ui.replayAction.disabled, false);
  assert.equal(ui.stackAction.disabled, true);

  current.processes.replay = processStatus({ exit_code: 1 });
  current.replay = {
    running: false,
    file_name: "capture.lcmlog",
    outcome: "failed",
    exit_code: 1,
    last_output: "lcm-logplayer exited with code 1.",
  };
  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Failed");
  assert.equal(ui.replayError.hidden, false);
  assert.equal(ui.replayError.textContent, "lcm-logplayer exited with code 1.");
});

test("operator-stopped replay is distinct from natural completion", () => {
  const ui = elements();
  const current = snapshot();

  current.replay = {
    running: false,
    file_name: "capture.lcmlog",
    outcome: "stopped",
    exit_code: -2,
    last_output: null,
  };
  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Stopped");
  assert.match(ui.replayDetail.textContent, /operator/);
  assert.equal(ui.replayError.hidden, true);

  current.replay.outcome = "completed";
  current.replay.exit_code = 0;
  render(current, new Set(), ui);
  assert.equal(ui.replayStatus.textContent, "Complete");
});
