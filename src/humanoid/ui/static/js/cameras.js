import { API, HttpKey } from "./constants.js";

export async function fetchCameraPreviews(depthFar) {
  const query = new URLSearchParams({ [HttpKey.DEPTH_FAR]: depthFar });
  const response = await fetch(`${API.CAMERA_PREVIEWS}?${query}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(4000),
  });
  if (!response.ok) throw new Error("Camera previews are unavailable.");
  return response.json();
}

export function renderCameraPreviews(cameras, elements) {
  const entries = Object.entries(cameras ?? {});
  const empty = entries.length === 0;
  elements.message.hidden = !empty;
  elements.depthKey.hidden = empty;
  elements.range.disabled = empty;
  if (empty) {
    elements.grid.replaceChildren();
    elements.message.textContent = cameras
      ? "No cameras configured for this robot."
      : "Camera previews unavailable.";
    return;
  }

  let index = 0;
  for (const [name, snapshot] of entries) {
    for (const stream of ["rgb", "depth"]) {
      const label = name + (stream === "rgb" ? " · RGB" : " · Native depth");
      let figure = elements.grid.children[index++];
      if (figure?.dataset.label !== label) {
        const next =
          elements.template.content.firstElementChild.cloneNode(true);
        next.dataset.label = label;
        next.querySelector("h3").textContent = label;
        next.querySelector("img").alt = label;
        if (figure) figure.replaceWith(next);
        else elements.grid.append(next);
        figure = next;
      }
      renderFrame(figure, snapshot[stream]);
    }
  }
  while (elements.grid.children.length > index)
    elements.grid.lastElementChild.remove();
  const scale = entries[0][1];
  elements.near.textContent = `≤ ${scale.depth_near_meters} m`;
  elements.far.textContent = `≥ ${scale.depth_far_meters} m`;
}

function renderFrame(figure, frame) {
  const image = figure.querySelector("img");
  const empty = figure.querySelector(".camera-image-area p");
  const status = figure.querySelector(".mini-status");
  const detail = figure.querySelector(".camera-detail");
  const live = Boolean(frame.connected && frame.image);
  const stale = !live && frame.age_seconds != null;
  image.hidden = !live;
  empty.hidden = live;
  status.textContent = live ? "Live" : stale ? "Stale" : "Waiting";
  status.classList.toggle("live", live);
  status.classList.toggle("stale", stale);
  if (live) {
    if (image.getAttribute("src") !== frame.image) image.src = frame.image;
    detail.textContent = `${frame.width} × ${frame.height} · ${frame.frame_id} · ${frame.age_seconds.toFixed(1)} s ago`;
  } else {
    image.removeAttribute("src");
    empty.textContent = stale ? "Camera stream stopped" : "Waiting for images";
    detail.textContent = stale
      ? `Last image ${frame.age_seconds.toFixed(1)} s ago`
      : "";
  }
}

export function startCameraPreviews(elements) {
  let pending = false;
  async function refresh() {
    if (pending || document.hidden) return;
    pending = true;
    const range = elements.range.value;
    try {
      const snapshot = await fetchCameraPreviews(range);
      // A range change while loading must not relabel an image using the old scale.
      if (range === elements.range.value)
        renderCameraPreviews(snapshot, elements);
    } catch {
      renderCameraPreviews(null, elements);
    } finally {
      pending = false;
    }
  }
  elements.range.addEventListener("change", refresh);
  document.addEventListener("visibilitychange", refresh);
  void refresh();
  window.setInterval(refresh, 200);
}
