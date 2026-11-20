import assert from "node:assert/strict";
import test from "node:test";
import { fetchCameraPreviews, renderCameraPreviews } from "../../../src/humanoid/ui/static/js/cameras.js";

function element() {
  return {
    hidden: false, textContent: "", src: null,
    classList: { toggle() {} },
    getAttribute(name) { return this[name]; },
    removeAttribute(name) { this[name] = null; },
  };
}
function elements() {
  const view = () => ({ image: element(), empty: element(), status: element(), detail: element() });
  return { rgb: view(), depth: view(), near: element(), far: element() };
}
const live = { connected: true, image: "data:image/jpeg;base64,pixels", width: 640, height: 480,
  frame_id: "optical", age_seconds: .1 };

test("RGB and native depth have independent live/waiting states and a metric scale", () => {
  const els = elements();
  renderCameraPreviews({ rgb: live, depth: { connected: false }, depth_near_meters: .2, depth_far_meters: 5 }, els);
  assert.equal(els.rgb.image.src, live.image);
  assert.equal(els.rgb.image.hidden, false);
  assert.equal(els.rgb.status.textContent, "Live");
  assert.equal(els.depth.status.textContent, "Waiting");
  assert.equal(els.depth.image.hidden, true);
  assert.equal(els.far.textContent, "≥ 5 m");
  assert.match(els.rgb.detail.textContent, /640 × 480/);
});

test("stale and disconnected previews remove old images rather than appearing live", () => {
  const els = elements();
  renderCameraPreviews({ rgb: live, depth: live }, els);
  renderCameraPreviews({ rgb: { connected: false, age_seconds: 3 }, depth: live }, els);
  assert.equal(els.rgb.status.textContent, "Stale");
  assert.equal(els.rgb.image.src, null);
  assert.equal(els.depth.status.textContent, "Live");
  renderCameraPreviews(null, els);
  assert.equal(els.depth.status.textContent, "Unavailable");
  assert.equal(els.depth.image.hidden, true);
  assert.equal(els.depth.image.src, null);
});

test("camera polling uses its own endpoint, range and bounded request", async () => {
  const original = globalThis.fetch;
  let request;
  globalThis.fetch = async (...args) => { request = args; return { ok: true, json: async () => ({ rgb: live }) }; };
  try {
    assert.deepEqual(await fetchCameraPreviews(10), { rgb: live });
    assert.equal(request[0], "/api/camera-previews?depth_far=10");
    assert.equal(request[1].cache, "no-store");
    assert.ok(request[1].signal instanceof AbortSignal);
  } finally { globalThis.fetch = original; }
});

test("overlays show source snapshots, switch to live immediately, and clear on stop", () => {
  const els = elements();
  els.overlays = { checked: true };
  els.summary = element();
  const annotated = { ...live, image: "annotated", frame_number: 12, age_seconds: 8 };
  const snapshot = { rgb: live, depth: live, inference: {
    rgb: annotated, depth: { ...annotated, frame_number: 13 },
    object_description: "toy car <img>", mask_count: 1, grasp_count: 2, selected: true,
  }};
  renderCameraPreviews(snapshot, els);
  assert.equal(els.rgb.image.src, "annotated");
  assert.equal(els.rgb.status.textContent, "Inference snapshot");
  assert.match(els.depth.detail.textContent, /Frame 13 · captured 8.0 s ago/);
  assert.match(els.summary.textContent, /toy car <img>/);
  els.overlays.checked = false;
  renderCameraPreviews(snapshot, els);
  assert.equal(els.rgb.image.src, live.image);
  assert.equal(els.rgb.status.textContent, "Live");
  els.overlays.checked = true;
  renderCameraPreviews({ rgb: live, depth: live, inference: null }, els);
  assert.equal(els.rgb.image.src, live.image);
  renderCameraPreviews(null, els);
  assert.equal(els.rgb.image.src, null);
  assert.equal(els.rgb.status.textContent, "Unavailable");
});
