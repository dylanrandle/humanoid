/* Keep Vuer input tied to fresh, visible WebXR frames and a server clock bound. */
(() => {
  const url = new URL(location.href);
  url.searchParams.set("frameloop", "always");
  url.searchParams.set(
    "ws",
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  );
  history.replaceState(null, "", url);
  // Vuer's pinned client resolves controller profiles/models from this CDN by
  // default. Keep its standard loader, but serve those exact assets locally.
  const nativeFetch = window.fetch.bind(window);
  const profileCdn =
    "https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0/dist/profiles/";
  window.fetch = (input, options) => {
    const address = input instanceof Request ? input.url : String(input);
    if (!address.startsWith(profileCdn)) return nativeFetch(input, options);
    const local = new URL(
      `/quest-assets/xr-profiles/${address.slice(profileCdn.length)}`,
      location.origin,
    );
    return nativeFetch(
      input instanceof Request ? new Request(local, input) : local,
      options,
    );
  };
  const sockets = new Set();
  const NativeSocket = window.WebSocket;
  let xrSession = null;
  let lastFrame = -Infinity;
  let lastSend = 0;
  const closeInputs = () => {
    for (const socket of sockets)
      socket.close(1000, "WebXR paused; release grips to reconnect");
  };
  const supportedController = (source) =>
    source.profiles.includes("meta-quest-touch-plus");
  const restrictControllers = (session) => {
    // Quest can expose emulated hand controllers even without hand-tracking.
    // Keep unsupported profiles out of Vuer's model loader and control state.
    const sources = Object.getOwnPropertyDescriptor(
      XRSession.prototype,
      "inputSources",
    ).get;
    Object.defineProperty(session, "inputSources", {
      get: () => Array.from(sources.call(session)).filter(supportedController),
    });
    session.addEventListener("inputsourceschange", (event) => {
      for (const key of ["added", "removed"])
        Object.defineProperty(event, key, {
          value: Array.from(event[key]).filter(supportedController),
        });
    });
  };
  window.WebSocket = class extends NativeSocket {
    constructor(address, protocols) {
      super(address, protocols);
      if (new URL(address, location.href).pathname !== "/ws") return;
      sockets.add(this);
      this.addEventListener("close", () => sockets.delete(this));
      this.addEventListener("message", (event) => {
        if (typeof event.data !== "string") return;
        event.stopImmediatePropagation();
        const message = JSON.parse(event.data);
        if (message.kind === "clock") {
          super.send(
            JSON.stringify({
              kind: "clock",
              nonce: message.nonce,
              client_ms: Date.now(),
            }),
          );
        }
        if (message.kind === "status") {
          const status = document.querySelector("#quest-status");
          if (status)
            status.textContent = message.tracking
              ? message.engaged
                ? "Quest connected · grip held"
                : "Tracking ready · hold a grip to control"
              : "Release both grips to arm · check controller tracking";
        }
      });
    }
    send(data) {
      if (
        xrSession &&
        (xrSession.visibilityState !== "visible" ||
          performance.now() - lastFrame > 150)
      ) {
        this.close(1000, "Tracking paused");
        return;
      }
      super.send(data);
    }
  };
  if (navigator.xr) {
    for (const method of ["requestSession", "offerSession"]) {
      if (typeof navigator.xr[method] !== "function") continue;
      const request = navigator.xr[method].bind(navigator.xr);
      navigator.xr[method] = async (mode, options = {}) => {
        // Physical controllers are required; optional hand models use another CDN.
        const session = await request(mode, {
          ...options,
          optionalFeatures: [
            ...new Set([
              ...(options.optionalFeatures || []).filter(
                (feature) => feature !== "hand-tracking",
              ),
              "layers",
            ]),
          ],
        });
        restrictControllers(session);
        xrSession = session;
        lastFrame = performance.now();
        session.addEventListener("visibilitychange", () => {
          if (session.visibilityState !== "visible") closeInputs();
        });
        session.addEventListener("end", () => {
          xrSession = null;
          closeInputs();
        });
        const referenceSpace = await session.requestReferenceSpace("local");
        const tick = (_time, frame) => {
          lastFrame = performance.now();
          const tracked = new Set();
          for (const source of session.inputSources) {
            if (
              source.gripSpace &&
              frame.getPose(source.gripSpace, referenceSpace)
            )
              tracked.add(source.handedness);
          }
          if (
            session.visibilityState === "visible" &&
            tracked.has("left") &&
            tracked.has("right") &&
            lastFrame - lastSend >= 15
          ) {
            const now = Date.now();
            for (const socket of sockets) {
              if (socket.readyState === NativeSocket.OPEN) {
                socket.send(
                  JSON.stringify({
                    kind: "frame",
                    client_ms: now,
                    frame_ms: now,
                  }),
                );
              }
            }
            lastSend = lastFrame;
          }
          session.requestAnimationFrame(tick);
        };
        session.requestAnimationFrame(tick);
        return session;
      };
    }
  }
  window.addEventListener("pagehide", closeInputs);
  document.addEventListener("DOMContentLoaded", () => {
    const status = document.createElement("div");
    status.id = "quest-status";
    status.textContent =
      "Enter VR or passthrough, then release both grips to arm";
    Object.assign(status.style, {
      position: "fixed",
      bottom: "14px",
      left: "14px",
      zIndex: "9999",
      color: "white",
      background: "#152531",
      padding: "10px",
      borderRadius: "8px",
      pointerEvents: "none",
    });
    document.body.append(status);
  });
})();
