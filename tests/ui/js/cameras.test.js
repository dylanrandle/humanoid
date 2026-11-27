import assert from "node:assert/strict";
import test from "node:test";
import { fetchCameras, needsCanvasPresentation, startCanvasPresentation, startCameraStreams, waitForIce } from "../../../src/humanoid/ui/static/js/cameras.js";

class Element extends EventTarget {
  constructor() { super(); this.hidden = false; this.textContent = ""; this.value = "3"; }
  classList = { toggle() {} };
  setAttribute(name, value) { this[name] = value; }
}
function elements() {
  const grid = { children: [], append(node) { this.children.push(node); }, replaceChildren() { this.children = []; } };
  function figure() {
    const nodes = new Map();
    return { querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, new Element());
      return nodes.get(selector);
    } };
  }
  return { grid, template: { content: { firstElementChild: { cloneNode: figure } } },
    range: new Element(), depthKey: new Element(), message: new Element(), near: new Element(), far: new Element() };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test("canvas workaround targets WebKit rather than Chromium or Firefox", () => {
  assert.equal(needsCanvasPresentation("Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15"), true);
  assert.equal(needsCanvasPresentation("Mozilla/5.0 AppleWebKit/537.36 Chrome/131.0 Safari/537.36"), false);
  assert.equal(needsCanvasPresentation("Mozilla/5.0 Gecko/20100101 Firefox/132.0"), false);
});

test("canvas displays decoded frames, resizes, hides when unavailable, and stops on cleanup", (t) => {
  const original = globalThis.window;
  t.after(() => { globalThis.window = original; });
  let tick, cleared, draws = 0, fail = false;
  globalThis.window = {
    setInterval(callback, delay) { tick = callback; assert.ok(delay >= 60); return 42; },
    clearInterval(id) { cleared = id; },
  };
  const video = { hidden: false, paused: false, readyState: 0, videoWidth: 0, videoHeight: 0 };
  const canvas = { hidden: true, getContext: () => ({ drawImage(source) {
    assert.equal(source, video);
    if (fail) throw new Error("Stream reset");
    draws += 1;
  } }) };
  const stop = startCanvasPresentation(video, canvas);
  tick();
  assert.equal(draws, 0);
  Object.assign(video, { readyState: 4, videoWidth: 640, videoHeight: 480 });
  tick();
  assert.equal(draws, 1);
  assert.equal(canvas.hidden, false);
  assert.equal(canvas.width, 640);
  assert.equal(canvas.height, 480);
  video.videoWidth = 320; video.videoHeight = 240;
  tick();
  assert.equal(canvas.width, 320);
  assert.equal(canvas.height, 240);
  video.paused = true;
  tick();
  assert.equal(canvas.hidden, true);
  assert.equal(draws, 2);
  video.paused = false; fail = true;
  assert.doesNotThrow(tick);
  assert.equal(canvas.hidden, true);
  fail = false;
  tick();
  assert.equal(canvas.hidden, false);
  video.hidden = true;
  tick();
  assert.equal(canvas.hidden, true);
  stop();
  assert.equal(cleared, 42);
});

test("camera metadata requests are bounded and never fetch image previews", async (t) => {
  let request;
  t.mock.method(globalThis, "fetch", async (...args) => { request = args; return { ok: true, json: async () => ({ cameras: ["base"] }) }; });
  assert.deepEqual(await fetchCameras(), { cameras: ["base"] });
  assert.equal(request[0], "/api/cameras");
  assert.equal(request[1].cache, "no-store");
  assert.ok(request[1].signal instanceof AbortSignal);
});

test("ICE gathering waits for complete candidates", async () => {
  const peer = new EventTarget();
  peer.iceGatheringState = "gathering";
  let complete = false;
  const waiting = waitForIce(peer).then(() => { complete = true; });
  await settle();
  assert.equal(complete, false);
  peer.iceGatheringState = "complete";
  peer.dispatchEvent(new Event("icegatheringstatechange"));
  await waiting;
  assert.equal(complete, true);
});

test("WebRTC views reconnect on range/config changes and release peers on hide/close", async (t) => {
  const original = { window: globalThis.window, document: globalThis.document, RTCPeerConnection: globalThis.RTCPeerConnection };
  const doc = new EventTarget(); doc.hidden = false;
  const win = new EventTarget(); let refresh;
  win.setInterval = (callback) => { refresh = callback; return 1; };
  win.clearInterval = () => {};
  const peers = [], requests = [];
  class Peer extends EventTarget {
    iceGatheringState = "complete";
    connectionState = "new";
    constructor(options) { super(); assert.deepEqual(options, { iceServers: [] }); peers.push(this); }
    addTransceiver(kind, options) { assert.equal(kind, "video"); assert.equal(options.direction, "recvonly"); }
    async createOffer() { return { type: "offer", sdp: "browser-sdp" }; }
    async setLocalDescription(description) { this.localDescription = description; }
    async setRemoteDescription(description) { assert.equal(description.type, "answer"); }
    close() { this.connectionState = "closed"; }
  }
  Object.assign(globalThis, { window: win, document: doc, RTCPeerConnection: Peer });
  t.after(() => Object.assign(globalThis, original));
  let config = { generation: 0, cameras: ["base", "wrist"], depth_near: 0.2, streams: {} };
  let id = 0;
  t.mock.method(globalThis, "fetch", async (path, options = {}) => {
    requests.push([path, options]);
    return { ok: true, json: async () => path.endsWith("/cameras") ? config : { session_id: String(++id), type: "answer", sdp: "backend-sdp" } };
  });
  const els = elements();
  const close = startCameraStreams(els);
  t.after(close);
  await settle();
  assert.deepEqual(els.grid.children.map(f => f.querySelector("h3").textContent),
    ["base · RGB", "base · Native depth", "wrist · RGB", "wrist · Native depth"]);
  assert.equal(peers.length, 4);
  assert.equal(requests.filter(([path]) => path.endsWith("/streams")).length, 4);
  assert.equal(JSON.parse(requests.find(([path]) => path.endsWith("/streams"))[1].body).depth_far, 3);
  config.streams["1"] = { connection_state: "connected", frame_age_seconds: 0.1 };
  const video = els.grid.children[0].querySelector("video");
  video.paused = true;
  video.readyState = 0;
  video.srcObject = { getTracks: () => [] };
  let plays = 0;
  video.play = async () => {
    plays += 1;
    assert.equal(video.hidden, false, "reveal video before requesting playback");
    throw new Error("Autoplay blocked");
  };
  await refresh();
  await settle();
  assert.equal(video.hidden, false);
  assert.equal(video.controls, true);
  await refresh();
  assert.equal(plays, 1, "do not repeatedly request blocked autoplay");
  assert.equal(els.grid.children[0].querySelector(".mini-status").textContent, "Paused");
  video.paused = false;
  video.readyState = 2;
  video.dispatchEvent(new Event("playing"));
  await refresh();
  assert.equal(video.controls, false);
  assert.equal(els.grid.children[0].querySelector(".mini-status").textContent, "Live");
  els.range.value = "5";
  els.range.dispatchEvent(new Event("change"));
  await settle();
  assert.ok(peers.slice(0, 4).every(p => p.connectionState === "closed"));
  assert.equal(peers.length, 8);
  doc.hidden = true; doc.dispatchEvent(new Event("visibilitychange"));
  assert.equal(els.grid.children.length, 0);
  assert.ok(peers.every(p => p.connectionState === "closed"));
  doc.hidden = false;
  config = { ...config, generation: 1, cameras: [] };
  doc.dispatchEvent(new Event("visibilitychange"));
  await settle();
  assert.match(els.message.textContent, /No cameras/);
  assert.equal(els.depthKey.hidden, true);
  close();
  assert.equal(requests.filter(([, options]) => options.method === "DELETE").length, 8);
});
