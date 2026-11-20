import { ApiRoute, Mode, ProcessName, Runtime } from "./constants.js";
import { processRequest, safetyPayload } from "./requests.js";

export function pickRequest(snapshot, description, confirmHardware) {
  if (!snapshot || !description.trim()) return null;
  const hardware = snapshot.runtime === Runtime.REAL;
  if (
    hardware &&
    !confirmHardware(
      "Run this pick on the real robot? Confirm the workspace is clear.",
    )
  )
    return null;
  return {
    path: ApiRoute.PICK_RUN,
    payload: {
      object: description.trim(),
      ...safetyPayload(snapshot, hardware),
    },
  };
}

export function renderPick(snapshot, busy, elements) {
  if (!elements) return;
  const process = snapshot.processes[ProcessName.PICK];
  const stackReady =
    snapshot.processes[ProcessName.STACK].running &&
    snapshot.orchestrator.connected;
  const running = process?.running ?? false;
  const status = snapshot.pick;
  const requestBusy =
    busy.has(ProcessName.PICK) ||
    busy.has(ApiRoute.PICK_RUN) ||
    busy.has(ApiRoute.PICK_CANCEL);
  elements.process.textContent = running
    ? "Stop pick policy"
    : "Start pick policy";
  elements.process.disabled =
    requestBusy || (!running && (!stackReady || snapshot.robot !== "triskel"));
  elements.object.disabled = !running || Boolean(status?.busy);
  elements.run.disabled =
    requestBusy ||
    !stackReady ||
    !running ||
    !status?.models_ready ||
    status.busy ||
    snapshot.orchestrator.mode !== Mode.IDLE ||
    !elements.object.value.trim();
  elements.cancel.disabled = requestBusy || !running || !status?.busy;
  elements.status.textContent =
    process?.last_output ||
    status?.message ||
    "Start the pick policy to load models";
}

export function bindPick(state, elements, perform, post, rerender) {
  elements.object.addEventListener("input", rerender);
  elements.process.addEventListener("click", () => {
    const request = processRequest(
      state.snapshot,
      ProcessName.PICK,
      window.confirm,
    );
    if (request)
      perform(ProcessName.PICK, () => post(request.path, request.payload));
  });
  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (elements.run.disabled) return;
    const request = pickRequest(
      state.snapshot,
      elements.object.value,
      window.confirm,
    );
    if (request)
      perform(ApiRoute.PICK_RUN, () => post(request.path, request.payload));
  });
  elements.cancel.addEventListener("click", () =>
    perform(ApiRoute.PICK_CANCEL, () => post(ApiRoute.PICK_CANCEL)),
  );
}
