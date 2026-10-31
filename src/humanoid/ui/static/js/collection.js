import { BusyKey, CollectionRunState } from "./constants.js";
import { collectionRunning, collectionUnavailable } from "./requests.js";

export function renderCollection(snapshot, busy, elements) {
  const fields = elements.collectionFields;
  const tasks = snapshot.collection_tasks || [];
  const taskSelect = fields.task;
  if (
    tasks.map((task) => task.name).join() !==
    [...taskSelect.options].map((option) => option.value).join()
  ) {
    const selected = taskSelect.value;
    taskSelect.replaceChildren(
      ...tasks.map((task) => {
        const option = taskSelect.ownerDocument.createElement("option");
        option.value = task.name;
        option.textContent = task.robot
          ? `${task.label} · ${task.robot}`
          : task.label;
        return option;
      }),
    );
    taskSelect.value = tasks.some((task) => task.name === selected)
      ? selected
      : tasks[0]?.name || "";
  }
  if (
    !elements.collectionForm.dataset.initialized &&
    snapshot.collection_defaults
  ) {
    for (const [name, value] of Object.entries(snapshot.collection_defaults))
      fields[name].value = String(value);
    if (
      snapshot.collection?.task &&
      tasks.some((task) => task.name === snapshot.collection.task)
    ) {
      fields.task.value = snapshot.collection.task;
      fields.seed.value = String(snapshot.collection.seed);
    }
    elements.collectionForm.dataset.initialized = "true";
  }
  const status = snapshot.collection;
  const active = collectionRunning(status);
  const preparing = status?.state === CollectionRunState.PREPARING;
  const pending = busy.has(BusyKey.COLLECTION);
  const task = tasks.find((task) => task.name === taskSelect.value);
  const unavailable = collectionUnavailable(snapshot, taskSelect.value);
  for (const [name, field] of Object.entries(fields))
    field.disabled =
      active ||
      pending ||
      (task?.collectable === false && !["task", "seed"].includes(name));
  elements.collectionAction.disabled =
    pending ||
    preparing ||
    (active
      ? status.state === CollectionRunState.STOPPING
      : Boolean(unavailable));
  elements.collectionAction.classList.toggle("stop", active);
  elements.collectionAction.textContent = pending
    ? "Working…"
    : preparing
      ? "Loading scene…"
      : status?.state === CollectionRunState.STOPPING
        ? "Stopping…"
        : active
          ? "Stop collection"
          : "Start collection";
  const labels = {
    idle: "Idle",
    preparing: "Loading scene",
    starting: "Starting",
    running: "Collecting",
    stopping: "Stopping",
    succeeded: "Complete",
    failed: "Failed",
    aborted: "Stopped",
  };
  elements.collectionStatus.textContent = labels[status?.state] || "Idle";
  elements.collectionDetail.textContent = preparing
    ? `Loading ${status.task || task?.label || "task"} · Seed ${status.seed}`
    : active
      ? `Running ${status.task || task?.label || "task"} · ${status.phase || "preparing"}${status.target ? ` · ${status.target}` : ""}`
      : unavailable ||
        "Task and seed changes load their scene while Idle. Only valid successes count.";
  elements.collectionProgress.textContent = status?.episodes
    ? `${status.successes} / ${status.episodes} episodes · Attempt ${status.attempts} / ${status.max_attempts} · Seed ${status.seed} · ${status.sim_seconds.toFixed(1)}s simulated · ${status.samples} samples`
    : "";
  elements.collectionRecording.hidden = !status?.recording;
  elements.collectionRecording.textContent = status?.recording
    ? `Recording: ${status.recording}`
    : "";
  elements.collectionMessage.hidden = !status?.message;
  elements.collectionMessage.textContent = status?.message || "";
  elements.collectionMessage.classList.toggle(
    "failed",
    status?.state === CollectionRunState.FAILED,
  );
}

export function disconnectCollection(elements) {
  for (const field of Object.values(elements.collectionFields))
    field.disabled = true;
  elements.collectionAction.disabled = true;
  elements.collectionStatus.textContent = "Unavailable";
  elements.collectionDetail.textContent =
    "Reconnect to view collection status.";
}
