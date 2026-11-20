import { API } from "./constants.js";

export async function fetchCameraPreviews(depthFar) {
  const response = await fetch(
    `${API.CAMERA_PREVIEWS}?depth_far=${encodeURIComponent(depthFar)}`,
    {
      cache: "no-store",
      signal: AbortSignal.timeout(4000),
    },
  );
  if (!response.ok) throw new Error("Camera previews are unavailable.");
  return response.json();
}

export function renderCameraPreviews(snapshot, elements) {
  const inference = elements.overlays?.checked ? snapshot?.inference : null;
  if (elements.summary) {
    elements.summary.textContent = inference
      ? `Inference snapshot for “${inference.object_description}” · ${inference.mask_count} mask${inference.mask_count === 1 ? "" : "s"} · ${inference.grasp_count} candidate${inference.grasp_count === 1 ? "" : "s"} shown. Cyan: segmentation; amber: generated grasps (not all executable); green: selected grasp. Uncheck Model overlays for live images.`
      : "Live images. Model overlays appear after running pickup.";
  }
  for (const stream of ["rgb", "depth"]) {
    const frame = inference?.[stream] ?? snapshot?.[stream];
    const view = elements[stream];
    const live = Boolean(frame?.connected && frame.image);
    const stale = !live && frame?.age_seconds != null;
    view.image.hidden = !live;
    view.empty.hidden = live;
    view.status.textContent = live
      ? inference
        ? "Inference snapshot"
        : "Live"
      : stale
        ? "Stale"
        : snapshot
          ? "Waiting"
          : "Unavailable";
    view.status.classList.toggle("live", live && !inference);
    view.status.classList.toggle("inference", Boolean(inference));
    view.status.classList.toggle("stale", stale);
    view.image.alt = `${inference ? "Annotated inference snapshot" : "Live camera view"}: ${stream === "rgb" ? "RGB" : "native depth"}`;
    if (live) {
      if (view.image.getAttribute("src") !== frame.image)
        view.image.src = frame.image;
      view.detail.textContent = `${frame.width} × ${frame.height} · ${frame.frame_id} · ${inference ? `Frame ${frame.frame_number} · captured ` : ""}${frame.age_seconds.toFixed(1)} s ago`;
    } else {
      view.image.removeAttribute("src");
      view.empty.textContent = stale
        ? "Camera stream stopped"
        : snapshot
          ? `Waiting for ${stream === "rgb" ? "RGB" : "depth"} images`
          : "Camera previews unavailable";
      view.detail.textContent = stale
        ? `Last image ${frame.age_seconds.toFixed(1)} s ago`
        : "Start a stack with a camera to view images.";
    }
  }
  if (snapshot) {
    elements.near.textContent = `≤ ${snapshot.depth_near_meters} m`;
    elements.far.textContent = `≥ ${snapshot.depth_far_meters} m`;
  }
}

export function startCameraPreviews(elements) {
  let pending = false;
  let latest = null;
  async function refresh() {
    if (pending || document.hidden) return;
    pending = true;
    const range = elements.range.value;
    try {
      const snapshot = await fetchCameraPreviews(range);
      // A range change while loading must not relabel an image using the old scale.
      if (range === elements.range.value) {
        latest = snapshot;
        renderCameraPreviews(snapshot, elements);
      }
    } catch {
      latest = null;
      renderCameraPreviews(null, elements);
    } finally {
      pending = false;
    }
  }
  elements.range.addEventListener("change", refresh);
  elements.overlays.addEventListener("change", () =>
    renderCameraPreviews(latest, elements),
  );
  document.addEventListener("visibilitychange", refresh);
  void refresh();
  window.setInterval(refresh, 200);
}
