import { fetchStatus, post } from "./api.js";
import {
  ApiRoute,
  DEFAULT_AGENT_MODEL,
  Mode,
  ProcessName,
} from "./constants.js";
import { safetyPayload } from "./requests.js";
import { showToast } from "./toast.js";

export function setupAgentPanel(getSnapshot) {
  const element = (name) => document.getElementById("agent-" + name);
  const form = element("form"),
    model = element("model"),
    task = element("task");
  const connect = element("connect"),
    run = element("run"),
    stop = element("stop");
  const login = element("login"),
    result = element("result");
  let agent = null,
    pending = false,
    generation = 0;

  function render() {
    const snapshot = getSnapshot();
    const mode = snapshot?.orchestrator.mode;
    const ready =
      snapshot?.robot === "triskel" &&
      snapshot.orchestrator.connected &&
      snapshot.processes[ProcessName.STACK].running &&
      snapshot.processes[ProcessName.AUTONOMY].running;
    const active = agent?.phase === "connecting" || agent?.phase === "running";
    const models = agent?.models ?? [];
    if (model.dataset.catalog !== JSON.stringify(models)) {
      const previous = model.value;
      model.replaceChildren(
        ...(models.length ? models : ["Connect to load models"]).map(
          (name) => new Option(name, models.length ? name : ""),
        ),
      );
      if (models.includes(previous)) model.value = previous;
      else if (models.includes(DEFAULT_AGENT_MODEL))
        model.value = DEFAULT_AGENT_MODEL;
      model.dataset.catalog = JSON.stringify(models);
    }
    if (active && agent.model) model.value = agent.model;
    connect.disabled = !agent || pending || active;
    connect.textContent = models.length ? "Refresh models" : "Connect ChatGPT";
    model.disabled = pending || active || !models.length;
    run.disabled =
      pending ||
      active ||
      !ready ||
      ![Mode.IDLE, Mode.AUTONOMY].includes(mode) ||
      !models.length ||
      !task.value.trim();
    task.disabled = pending || active;
    stop.hidden = !active;
    stop.disabled = pending;
    element("mode").textContent = !snapshot
      ? "Waiting for the console"
      : !ready
        ? "Select Triskel and start the stack and Agent API."
        : mode === Mode.IDLE
          ? "Idle · observation only"
          : mode === Mode.AUTONOMY
            ? "Autonomy · motion enabled"
            : "Select Idle or Autonomy.";
    login.hidden = !(
      agent?.login_code && agent?.login_url?.startsWith("https://")
    );
    if (!login.hidden) {
      login.querySelector("a").href = agent.login_url;
      login.querySelector("strong").textContent = agent.login_code;
    } else {
      login.querySelector("a").removeAttribute("href");
      login.querySelector("strong").textContent = "";
    }
    const message = agent?.message || "";
    const output = [agent?.instruction, message].filter(Boolean).join("\n\n");
    if (result.textContent !== output) result.textContent = output;
    result.hidden = !output;
  }

  async function refresh() {
    if (pending) return;
    const version = generation;
    try {
      const next = await fetchStatus(ApiRoute.AGENT);
      if (version === generation) agent = next;
    } catch {
      if (version === generation) agent = null;
    }
    render();
  }

  async function act(path, payload = {}) {
    if (pending) return;
    pending = true;
    generation++;
    render();
    try {
      agent = await post(path, payload);
    } catch (error) {
      showToast(error.message);
    } finally {
      pending = false;
      render();
    }
  }

  connect.addEventListener("click", () => act(ApiRoute.AGENT_CONNECT));
  stop.addEventListener("click", () => act(ApiRoute.AGENT_STOP));
  task.addEventListener("input", render);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const snapshot = getSnapshot();
    if (!snapshot || run.disabled) return;
    void act(ApiRoute.AGENT_RUN, {
      instruction: task.value.trim(),
      model: model.value,
      mode: snapshot.orchestrator.mode,
      ...safetyPayload(snapshot),
    });
  });
  return { refresh, render };
}
