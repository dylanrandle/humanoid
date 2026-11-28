# Quest headset

Use a Meta Quest 3 and its physical Touch Plus controllers to teleoperate real or
simulated robots through Vuer. The viewer displays the robot's configured cameras
as a main image and an optional top-right inset. On Triskel, these are the base and
wrist cameras.

## Connections

For wireless access on Triskel, use the [local HTTPS gateway](../gateway.md).
On a Mac, run the simulator and dashboard on demand and use USB for the viewer and
controller connection. The Mac workflow needs no Caddy, persistent services,
certificates, or `.local` discovery.

## Mac USB

From the repository root, start the backend in a terminal:

```bash
uv run start --no-open
```

Use `http://localhost:8765` for the Mac dashboard. Leave `HUMANOID_PUBLIC_HOST`
unset and omit `--public-host` for this workflow. Select simulation, choose the
scene, then start the stack and Quest input when ready.

Connect a USB data cable, enable Developer Mode on the headset, and accept its USB
debugging prompt. Install Android Platform Tools; the helper finds `adb` on `PATH`
or in the standard Mac Android SDK directory. In another terminal:

```bash
./scripts/forward-quest start
```

Open `http://localhost:8012/connect` in the Quest browser, choose **Open Vuer**, and
enter VR. The helper forwards both Vuer and the dashboard, prints their headset
URLs, and exits while the mappings remain active. It selects the USB device using
`adb -d`; connect only one USB Android device. Rerun it after reconnecting the cable.

`HUMANOID_QUEST_PORT` controls the Quest listener and helper mapping; use the same
value for both. `HUMANOID_DASHBOARD_PORT` controls the helper's dashboard mapping
and must match the backend's `--port` argument if customized. The helper does not
start the backend. The connection page's dashboard link follows the running
backend's port. The helper uses `uv run` to read the shared `Host` and `Port` enums in
[constants.py](../../src/humanoid/constants.py), using the same defaults as the
backend, so run it from a checkout with the project environment available.
Use `./scripts/forward-quest --help` for usage.

Meta documents [USB port forwarding for Browser](https://developers.meta.com/vr/documentation/web/browser-remote-debugging/)
and confirms that [forwarded HTTP localhost is a secure context for WebXR](https://github.com/meta-quest/iwsdk-replit-template#iterate-against-a-laptop-tethered-headset).
Vuer still reads browser controller events; ADB only provides the USB tunnel.
The composed camera view has been checked on a Quest 3 over Mac USB in passthrough.

For controller operation and camera layout, see
[Quest teleoperation](../operation.md#quest-teleoperation).

## Camera transport

Keep the Quest and Mac on the same Wi-Fi network for camera feeds while using USB
for controller transport. ADB reverse carries the page and signaling; it does not
carry the UDP video. No Mac HTTPS gateway is needed. See
[Networking](../networking.md#camera-streaming) for the media path.

## Stop the session

Stop input and stack in the dashboard, then press Ctrl-C in the backend terminal.
Remove the mappings that you created:

```bash
./scripts/forward-quest stop
```

Use the same port overrides when stopping if you customized them. The helper
removes only these two mappings. No login service restarts the backend; start it
explicitly for the next session.

## Architecture

`BackendApplication.start()` starts the loopback Quest server alongside supervision;
`close()` shuts it down before releasing camera and streaming services. The latest
controller snapshot belongs to the same application. The headset connects through
Triskel HTTPS or the Mac USB tunnel. Input freshness checks and
neutral rearming pause commands after tracking loss or runtime changes.
Automated simulation trials set `serve_quest=False` to omit the headset listener
from their isolated backends.

The device integration lives in
[hardware/teleop/quest](../../src/humanoid/hardware/teleop/quest/). Its
[server](../../src/humanoid/hardware/teleop/quest/server.py) manages the listener thread
and event loop. Its `create_app` factory builds the headset application, serves
assets, and adapts HTTP, WebSocket, and video-signaling requests. Each application
owns a session and closes it during shutdown.
[QuestSession](../../src/humanoid/hardware/teleop/quest/session.py) owns the active
viewer connection, controller protocol, scene updates, and video-peer lifecycle.
It reserves ownership before the WebSocket upgrade and releases it on failed
handshakes or disconnects. The
[input state](../../src/humanoid/hardware/teleop/quest/input.py) separately owns freshness
and neutral rearming, using the [wire types](../../src/humanoid/types/quest.py).

The integration does not depend on `BackendApplication`. The backend supplies its
configuration, input state, and typed camera/streaming callbacks, including a
consistent snapshot of camera readers and their runtime generation. Shared camera
and streaming services remain lazy and backend-owned. The
[dashboard Quest routes](../../src/humanoid/api/routes/quest.py) expose validated
input to the policy worker and link to the headset viewer.

Starting or stopping the Quest policy worker does not start or stop the headset
server. Keeping the server with the backend lets the viewer remain connected
while teleoperation input is stopped or restarted.

The teleoperation worker uses
[QuestClient](../../src/humanoid/api/clients/quest.py) to poll validated
backend snapshots on a background thread. The
[policy](../../src/humanoid/policy/teleop/quest.py) consumes those snapshots without
waiting on HTTP and converts them into robot targets; the
[node](../../src/humanoid/nodes/policy/teleop/quest.py) runs the policy and publishes
its actions through the realtime environment.

The [camera rendering utility](../../src/humanoid/utils/camera_rendering.py)
combines a primary image and an inset. The Quest session uses the robot's camera
configuration order, filtered to cameras supported by the selected runtime: the
first camera fills the main view and the second supplies the inset. A single camera
fills the view without an inset; additional cameras are not shown in this layout.
Triskel configures its base camera first and wrist camera second. The session
supplies the readers, layout settings, and camera names as labels, then uses the
shared streaming service to deliver one headset stream and releases its peers on
disconnect.

The viewer serves Vuer's original JavaScript assets. Panel placement uses Vuer's
position and rotation properties; no rendering components are replaced. The
browser input guard remains responsible for tracking freshness and supported
controller profiles.

## Controller assets

The viewer bundles the Meta Quest 3 Touch Plus controller profile and models
so loading these assets does not require internet access. The browser guard
redirects Vuer's profile requests to these same-origin assets. Physical controllers
are required. The guard excludes optional hand tracking for both explicitly
requested and browser-offered XR sessions. It also filters emulated hand inputs
and other unsupported profiles before Vuer loads models or reads controller state.

### Source and updates

The files are a subset of `@webxr-input-profiles/assets` version **1.0.20**, from
the [npm release archive](https://registry.npmjs.org/@webxr-input-profiles/assets/-/assets-1.0.20.tgz).
Only `meta-quest-touch-plus`, its left/right models, and its entry in
`profilesList.json` are included. The `meta-quest-touch-plus-v2` profile is for the
[Quest 3S Xbox Edition](https://github.com/immersive-web/webxr-input-profiles/pull/276)
and is not bundled.

The profile and models retain their upstream contents, including fallback IDs in
the profile metadata. Those IDs do not require bundling other models: Vuer selects
the matching profile from the headset's reported list. Preserve the included
license when updating the assets.

## See also

- [Documentation index](../README.md)
- [Hardware overview](README.md)
- [Operating the robot](../operation.md#quest-teleoperation)
- [Robot backend](../api.md)
- [Local HTTPS gateway](../gateway.md)
- [Quest configuration](../../src/humanoid/config/quest.py)
- [USB forwarding helper](../../scripts/forward-quest)
- [Bundled controller assets](../../src/humanoid/ui/quest/xr-profiles/)
- [Upstream asset license](../../src/humanoid/ui/quest/xr-profiles/LICENSE.md)
- [Browser guard](../../src/humanoid/ui/quest/guard.js)
