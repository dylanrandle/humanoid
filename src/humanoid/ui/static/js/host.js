export function renderHostResources(status, elements) {
  const cpu = status?.cpu_percent;
  const memory = status?.memory_percent;
  elements.cpu.textContent = Number.isFinite(cpu)
    ? `${cpu.toFixed(1)}%`
    : status
      ? "Sampling…"
      : "Unavailable";
  elements.memory.textContent = Number.isFinite(memory)
    ? `${memory.toFixed(1)}%`
    : "Unavailable";
  elements.memoryDetail.textContent = status
    ? `${gib(status.memory_used_bytes)} / ${gib(status.memory_total_bytes)} GiB used`
    : "Waiting for host telemetry";
  for (const [meter, value] of [
    [elements.cpuMeter, cpu],
    [elements.memoryMeter, memory],
  ]) {
    meter.hidden = !Number.isFinite(value);
    meter.value = Number.isFinite(value) ? value : 0;
  }
}

function gib(bytes) {
  return (bytes / 1024 ** 3).toFixed(1);
}
