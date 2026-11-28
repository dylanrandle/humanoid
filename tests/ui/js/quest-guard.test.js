import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(
  new URL("../../../src/humanoid/ui/quest/guard.js", import.meta.url),
  "utf8",
);

async function browser(method = "requestSession") {
  const requests = [];
  let now = 100;
  let draw;
  const events = new Map();
  class Session {
    visibilityState = "visible";
    sources = ["left", "right"].map((handedness) => ({
      handedness,
      profiles: ["meta-quest-touch-plus", "generic-trigger"],
      gripSpace: {},
    }));
    get inputSources() {
      return this.sources;
    }
    async requestReferenceSpace() {
      return {};
    }
    requestAnimationFrame(fn) {
      draw = fn;
    }
    addEventListener(name, fn) {
      events.set(name, fn);
    }
  }
  const session = new Session();
  class Socket {
    static OPEN = 1;
    readyState = 1;
    messages = [];
    events = new Map();
    addEventListener(name, fn) {
      this.events.set(name, fn);
    }
    send(message) {
      this.messages.push(message);
    }
    close() {
      this.readyState = 3;
      this.events.get("close")?.();
    }
  }
  const fetched = [];
  const window = {
    WebSocket: Socket,
    addEventListener() {},
    fetch(input) {
      fetched.push(input instanceof Request ? input.url : String(input));
    },
  };
  const context = {
    window,
    location: new URL("https://triskel.local:8443"),
    history: { replaceState() {} },
    URL,
    Request,
    XRSession: Session,
    performance: { now: () => now },
    navigator: {
      xr: Object.fromEntries(
        ["requestSession", "offerSession"].map((name) => [
          name,
          async (mode, options) => {
            requests.push({ name, mode, options });
            return session;
          },
        ]),
      ),
    },
    document: { addEventListener() {}, querySelector() {} },
  };
  vm.runInNewContext(source, context);
  const socket = new window.WebSocket("wss://triskel.local:8443/ws");
  await context.navigator.xr[method]("immersive-vr", {
    optionalFeatures: ["hand-tracking", "local-floor", "layers"],
  });
  return {
    socket,
    session,
    events,
    window,
    fetched,
    requests,
    draw(tracked = true) {
      now += 20;
      draw(now, { getPose: () => (tracked ? {} : null) });
    },
    advance(ms) {
      now += ms;
    },
  };
}

test("Quest pulses require native frames and both tracked controllers", async () => {
  const page = await browser();
  page.draw();
  assert.equal(JSON.parse(page.socket.messages.at(-1)).kind, "frame");
  const count = page.socket.messages.length;
  page.draw(false);
  page.session.sources.pop();
  page.draw();
  assert.equal(page.socket.messages.length, count);
  page.advance(200);
  page.socket.send(new Uint8Array([1]));
  assert.equal(page.socket.readyState, 3);
  assert.equal(page.socket.messages.length, count);
});

test("Quest visibility loss and session end close input transport", async () => {
  for (const event of ["visibilitychange", "end"]) {
    const page = await browser();
    page.session.visibilityState = "hidden";
    page.events.get(event)();
    assert.equal(page.socket.readyState, 3);
  }
});

test("Quest clock handshake stays out of Vuer's binary message handler", async () => {
  const page = await browser();
  let intercepted = false;
  page.socket.events.get("message")({
    data: JSON.stringify({ kind: "clock", nonce: "challenge" }),
    stopImmediatePropagation() {
      intercepted = true;
    },
  });
  const reply = JSON.parse(page.socket.messages.at(-1));
  assert.equal(intercepted, true);
  assert.equal(reply.nonce, "challenge");
  assert.ok(reply.client_ms > 0);
});

test("Quest profiles and model requests stay local, including Request objects", async () => {
  const page = await browser();
  const cdn =
    "https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0/dist/profiles/";
  await page.window.fetch(cdn + "profilesList.json");
  await page.window.fetch(new Request(cdn + "meta-quest-touch-plus/left.glb"));
  assert.deepEqual(page.fetched, [
    "https://triskel.local:8443/quest-assets/xr-profiles/profilesList.json",
    "https://triskel.local:8443/quest-assets/xr-profiles/meta-quest-touch-plus/left.glb",
  ]);
});

for (const method of ["requestSession", "offerSession"]) {
  test(`Quest ${method} disables hand tracking and installs the frame guard`, async () => {
    const page = await browser(method);
    assert.equal(page.requests[0].name, method);
    assert.deepEqual(Array.from(page.requests[0].options.optionalFeatures), [
      "local-floor",
      "layers",
    ]);
    page.draw();
    assert.equal(JSON.parse(page.socket.messages.at(-1)).kind, "frame");
    page.session.visibilityState = "hidden";
    page.events.get("visibilitychange")();
    assert.equal(page.socket.readyState, 3);
  });
}

test("Quest emulated hands are excluded from initial sources and source-change events", async () => {
  const page = await browser("offerSession");
  const controller = page.session.sources[0];
  const hand = {
    handedness: "right",
    profiles: ["meta-fixed-hand", "generic-hand-select"],
    gripSpace: {},
  };
  page.session.sources = [controller, hand];
  assert.deepEqual(Array.from(page.session.inputSources), [controller]);
  const event = { added: [hand, controller], removed: [hand] };
  page.events.get("inputsourceschange")(event);
  assert.deepEqual(Array.from(event.added), [controller]);
  assert.deepEqual(Array.from(event.removed), []);
  page.draw();
  assert.equal(page.socket.messages.length, 0);
});
