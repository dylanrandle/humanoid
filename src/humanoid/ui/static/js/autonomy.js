import { ApiRoute } from "./constants.js";
import { post } from "./api.js";
import { safetyPayload } from "./requests.js";

export function autonomyAvailability(snapshot, autonomy, busy) {
  const simulation = snapshot?.runtime === "sim";
  const stackRunning = snapshot?.processes?.stack?.running;
  return {
    select: Boolean(simulation && !stackRunning && !autonomy.running && !busy),
    start: Boolean(
      simulation &&
      autonomy.task &&
      autonomy.ready &&
      !autonomy.running &&
      !busy,
    ),
    stop: Boolean(autonomy.running),
  };
}

export function startAutonomyPanel(getSnapshot, showError) {
  const task = document.getElementById("autonomy-task");
  const start = document.getElementById("autonomy-start");
  const stop = document.getElementById("autonomy-stop");
  const status = document.getElementById("autonomy-status");
  let busy = false;
  let loading = false;
  let current = null;

  function render() {
    if (!current) return;
    const available = autonomyAvailability(getSnapshot(), current, busy);
    task.disabled = !available.select;
    start.disabled = !available.start;
    stop.disabled = !available.stop;
    status.textContent = current.message;
  }

  async function refresh() {
    if (loading) return;
    loading = true;
    try {
      const response = await fetch(ApiRoute.AUTONOMY, { cache: "no-store" });
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error || "Autonomy status unavailable.");
      current = data;
      const options = [
        new Option("Select a task", ""),
        ...data.tasks.map(
          (name) =>
            new Option(
              name === "toy_car_cleanup" ? "Collect toy cars" : name,
              name,
            ),
        ),
      ];
      task.replaceChildren(...options);
      task.value = data.task || "";
      render();
    } catch (error) {
      status.textContent = error.message;
      task.disabled = true;
      start.disabled = true;
      // Keep stop available after a status-fetch failure during an active run.
      stop.disabled = !current?.running;
    } finally {
      loading = false;
    }
  }

  async function action(path, payload) {
    if (busy && path !== "stop") return;
    busy = true;
    render();
    try {
      current = await post(
        {
          select: ApiRoute.AUTONOMY_SELECT,
          start: ApiRoute.AUTONOMY_START,
          stop: ApiRoute.AUTONOMY_STOP,
        }[path],
        payload,
      );
    } catch (error) {
      showError(error.message);
    } finally {
      busy = false;
      render();
      await refresh();
    }
  }

  task.addEventListener("change", () => {
    const snapshot = getSnapshot();
    if (snapshot && task.value)
      void action("select", { task: task.value, ...safetyPayload(snapshot) });
  });
  start.addEventListener("click", () => {
    const snapshot = getSnapshot();
    if (snapshot) void action("start", safetyPayload(snapshot));
  });
  stop.addEventListener("click", () => {
    void action("stop", {});
  });
  void refresh();
  return window.setInterval(() => {
    void refresh();
  }, 1000);
}
