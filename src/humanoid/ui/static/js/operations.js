export async function performOperation({
  state,
  key,
  action,
  renderSnapshot,
  showError,
  refresh,
}) {
  if (state.busy.has(key)) return false;
  state.busy.add(key);
  renderSnapshot();
  try {
    state.snapshot = await action();
    renderSnapshot();
  } catch (error) {
    showError(errorMessage(error));
  } finally {
    state.busy.delete(key);
    if (state.refreshPromise) await state.refreshPromise;
    await refresh();
  }
  return true;
}

export function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
