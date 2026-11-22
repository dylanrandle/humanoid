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
  const grid = {
    children: [],
    append(child) { this.children.push(child); },
    replaceChildren() { this.children.length = 0; },
    get lastElementChild() { return this.children.at(-1); },
  };
  function figure() {
    const nodes = new Map();
    return {
      dataset: {},
      querySelector(selector) {
        if (!nodes.has(selector)) nodes.set(selector, element());
        return nodes.get(selector);
      },
      replaceWith(node) { grid.children.splice(grid.children.indexOf(this), 1, node); },
      remove() { grid.children.splice(grid.children.indexOf(this), 1); },
    };
  }
  return { grid, template: { content: { firstElementChild: { cloneNode: figure } } },
    range: element(), depthKey: element(), message: element(), near: element(), far: element() };
}
const live = { connected: true, image: "data:image/jpeg;base64,pixels", width: 640, height: 480,
  frame_id: "optical", age_seconds: .1 };
const snapshot = (rgb, depth) => ({ rgb, depth, depth_near_meters: .2, depth_far_meters: 5 });

test("all named views render together and update independently", () => {
  const els = elements();
  const wristImage = { ...live, image: "data:image/jpeg;base64,wrist" };
  const wrist = snapshot(wristImage, { connected: false });
  renderCameraPreviews({ base: snapshot(live, live), wrist }, els);
  const [rgb, depth, wristRgb, wristDepth] = els.grid.children;
  assert.deepEqual(els.grid.children.map(f => f.querySelector("h3").textContent),
    ["base · RGB", "base · Native depth", "wrist · RGB", "wrist · Native depth"]);
  assert.equal(wristRgb.querySelector("img").src, wristImage.image);
  assert.equal(wristDepth.querySelector(".mini-status").textContent, "Waiting");
  assert.equal(els.far.textContent, "≥ 5 m");
  renderCameraPreviews({ base: snapshot({ connected: false, age_seconds: 3 }, live), wrist }, els);
  assert.equal(rgb.querySelector(".mini-status").textContent, "Stale");
  assert.equal(rgb.querySelector("img").src, null);
  assert.equal(depth.querySelector("img").src, live.image);
  renderCameraPreviews({ inspection: wrist }, els);
  assert.equal(els.grid.children.length, 2);
  assert.equal(els.grid.children[0].querySelector("h3").textContent, "inspection · RGB");
});

test("switching to a camera-free robot or losing the connection clears previews", () => {
  const els = elements();
  for (const [payload, expected] of [[{}, /No cameras/], [null, /unavailable/]]) {
    renderCameraPreviews({ base: snapshot(live, live) }, els);
    renderCameraPreviews(payload, els);
    assert.equal(els.grid.children.length, 0);
    assert.equal(els.depthKey.hidden, true);
    assert.match(els.message.textContent, expected);
  }
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
