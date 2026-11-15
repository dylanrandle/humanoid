import assert from "node:assert/strict";
import test from "node:test";
import { renderHostResources } from "../../../src/humanoid/ui/static/js/host.js";

function elements() {
  return Object.fromEntries(["cpu", "cpuMeter", "memory", "memoryMeter", "memoryDetail"].map(name => [name, {}]));
}
const sample = {
  cpu_percent: 25.5, memory_percent: 50,
  memory_used_bytes: 8 * 1024 ** 3, memory_total_bytes: 16 * 1024 ** 3,
};

test("host values show total CPU and used/total GiB", () => {
  const els = elements();
  renderHostResources(sample, els);
  assert.equal(els.cpu.textContent, "25.5%");
  assert.equal(els.cpuMeter.value, 25.5);
  assert.equal(els.memory.textContent, "50.0%");
  assert.equal(els.memoryDetail.textContent, "8.0 / 16.0 GiB used");
});

test("first sample and disconnected state never display a fake zero or stale value", () => {
  const els = elements();
  renderHostResources({ ...sample, cpu_percent: null }, els);
  assert.equal(els.cpu.textContent, "Sampling…");
  assert.equal(els.cpuMeter.hidden, true);
  assert.equal(els.memoryMeter.hidden, false);
  renderHostResources(sample, els);
  renderHostResources(null, els);
  assert.equal(els.cpu.textContent, "Unavailable");
  assert.equal(els.memory.textContent, "Unavailable");
  assert.equal(els.cpuMeter.hidden, true);
  assert.equal(els.memoryMeter.hidden, true);
});
