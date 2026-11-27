import { ApiRoute } from "./constants.js";

export async function fetchCameras() {
  const response = await fetch(ApiRoute.CAMERAS, {
    cache: "no-store",
    signal: AbortSignal.timeout(4000),
  });
  if (!response.ok) throw new Error("Camera information is unavailable.");
  return response.json();
}

export function waitForIce(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => finish(new Error("Camera connection timed out.")),
      12000,
    );
    function finish(error) {
      clearTimeout(timeout);
      peer.removeEventListener("icegatheringstatechange", changed);
      if (error) reject(error);
      else resolve();
    }
    function changed() {
      if (peer.iceGatheringState === "complete") finish();
    }
    peer.addEventListener("icegatheringstatechange", changed);
  });
}

export function needsCanvasPresentation(
  userAgent = globalThis.navigator?.userAgent ?? "",
) {
  return (
    /AppleWebKit/.test(userAgent) &&
    !/Chrome|Chromium|Edg|OPR|Android/.test(userAgent)
  );
}

export function startCanvasPresentation(video, canvas) {
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) return () => {};
  // Safari can decode a WebRTC frame (including drawImage) without presenting
  // it in the video layer. Keep that video in layout for autoplay and paint its
  // decoded frames above it. Do not depend on video presentation callbacks.
  const timer = window.setInterval(() => {
    if (
      video.hidden ||
      video.paused ||
      video.readyState < 2 ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      canvas.hidden = true;
      return;
    }
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    try {
      context.drawImage(video, 0, 0);
      canvas.hidden = false;
    } catch {
      // The stream can reset between the readiness check and drawing.
      canvas.hidden = true;
    }
  }, 1000 / 15);
  return () => {
    window.clearInterval(timer);
    canvas.hidden = true;
  };
}

export function startCameraStreams(elements) {
  const sessions = new Map();
  let generation = null;
  let pending = false;
  let stopped = false;
  let revision = 0;

  function closeSession(session) {
    session.closed = true;
    session.stopPresentation?.();
    session.peer.close();
    const video = session.figure.querySelector("video");
    video.srcObject?.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
    if (session.id) {
      void fetch(ApiRoute.STREAM.replace("<session_id>", session.id), {
        method: "DELETE",
        keepalive: true,
      }).catch(() => {});
    }
  }

  function reset() {
    revision += 1;
    for (const session of sessions.values()) closeSession(session);
    sessions.clear();
    elements.grid.replaceChildren();
  }

  async function connect(camera, stream, config) {
    const figure = elements.template.content.firstElementChild.cloneNode(true);
    const label = `${camera} · ${stream === "rgb" ? "RGB" : "Native depth"}`;
    figure.querySelector("h3").textContent = label;
    const video = figure.querySelector("video");
    video.setAttribute("aria-label", label);
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    elements.grid.append(figure);
    const peer = new RTCPeerConnection({ iceServers: [] });
    const session = {
      peer,
      figure,
      id: null,
      closed: false,
      failed: false,
      playPending: false,
      playbackBlocked: false,
      stopPresentation: needsCanvasPresentation()
        ? startCanvasPresentation(video, figure.querySelector("canvas"))
        : null,
    };
    video.addEventListener("playing", () => {
      session.playbackBlocked = false;
      video.controls = false;
    });
    sessions.set(`${camera}/${stream}`, session);
    peer.addTransceiver("video", { direction: "recvonly" });
    peer.ontrack = (event) => {
      if (!session.closed) {
        video.srcObject = new MediaStream([event.track]);
      }
    };
    peer.onconnectionstatechange = () => {
      if (["failed", "disconnected", "closed"].includes(peer.connectionState)) {
        session.failed = true;
      }
    };
    try {
      await peer.setLocalDescription(await peer.createOffer());
      await waitForIce(peer);
      if (session.closed) return;
      const response = await fetch(ApiRoute.STREAMS, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          camera,
          stream,
          depth_far: Number(elements.range.value),
          sdp: peer.localDescription.sdp,
          type: peer.localDescription.type,
        }),
        signal: AbortSignal.timeout(20000),
      });
      if (!response.ok) throw new Error("Camera connection failed.");
      const answer = await response.json();
      session.id = answer.session_id;
      if (session.closed) {
        closeSession(session);
        return;
      }
      await peer.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    } catch {
      if (!session.closed) {
        closeSession(session);
        session.failed = true;
        figure.querySelector(".camera-image-area p").textContent =
          "Reconnecting…";
      }
    }
  }

  async function refresh() {
    if (pending || stopped || document.hidden) return;
    pending = true;
    const requestedRevision = revision;
    try {
      const config = await fetchCameras();
      if (requestedRevision !== revision || stopped || document.hidden) return;
      if (
        generation !== config.generation ||
        [...sessions.values()].some((s) => s.failed)
      ) {
        reset();
        generation = config.generation;
      }
      const hasCameras = config.cameras.length > 0;
      elements.message.hidden = hasCameras;
      elements.message.textContent = "No cameras configured for this robot.";
      elements.depthKey.hidden = !hasCameras;
      elements.range.disabled = !hasCameras;
      elements.near.textContent = `≤ ${config.depth_near} m`;
      elements.far.textContent = `≥ ${elements.range.value} m`;
      for (const camera of config.cameras) {
        for (const stream of ["rgb", "depth"]) {
          if (!sessions.has(`${camera}/${stream}`))
            void connect(camera, stream, config);
        }
      }
      for (const session of sessions.values()) {
        const metrics = config.streams[session.id];
        const sourceReady =
          metrics?.connection_state === "connected" &&
          metrics.frame_age_seconds != null;
        const video = session.figure.querySelector("video");
        video.hidden = !sourceReady;
        // Start playback after revealing the video. A connected sender does not
        // imply the browser is playing, and autoplay rejection needs a fallback.
        if (
          sourceReady &&
          video.srcObject &&
          video.paused &&
          !session.playPending &&
          !session.playbackBlocked
        ) {
          session.playPending = true;
          void video
            .play()
            .catch(() => {
              if (!session.closed) {
                session.playbackBlocked = true;
                video.controls = true;
              }
            })
            .finally(() => {
              session.playPending = false;
            });
        }
        const live = sourceReady && !video.paused && video.readyState >= 2;
        const status = session.figure.querySelector(".mini-status");
        status.textContent = live
          ? "Live"
          : session.playbackBlocked
            ? "Paused"
            : "Waiting";
        status.classList.toggle("live", live);
        session.figure.querySelector(".camera-image-area p").hidden =
          sourceReady;
        session.figure.querySelector(".camera-detail").textContent = live
          ? `${metrics.frame_age_seconds.toFixed(1)} s ago`
          : session.playbackBlocked
            ? "Playback paused — press play on the video."
            : sourceReady
              ? "Waiting for video playback"
              : "Waiting for camera frames";
      }
    } catch {
      elements.message.hidden = false;
      elements.message.textContent =
        "Camera connection unavailable. Reconnecting…";
      reset();
    } finally {
      pending = false;
    }
  }

  const timer = window.setInterval(refresh, 1000);
  const rangeChanged = () => {
    reset();
    void refresh();
  };
  const visibilityChanged = () => {
    if (document.hidden) reset();
    else void refresh();
  };
  const close = () => {
    if (stopped) return;
    stopped = true;
    window.clearInterval(timer);
    reset();
    elements.range.removeEventListener("change", rangeChanged);
    document.removeEventListener("visibilitychange", visibilityChanged);
    window.removeEventListener("pagehide", close);
  };
  elements.range.addEventListener("change", rangeChanged);
  document.addEventListener("visibilitychange", visibilityChanged);
  window.addEventListener("pagehide", close);
  void refresh();
  return close;
}
