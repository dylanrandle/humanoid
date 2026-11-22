# Cameras

The real-hardware stack and replay start `CameraDriverNode` when the selected robot has a configured hardware camera.
Triskel binds its bottom-plate D435 and wrist D405 by serial number in
`config/robot/triskel.py`. The `base` stream uses `CAMERA/BASE/*` and the `wrist` stream uses
`CAMERA/WRIST/*` with the same message types. The wrist driver retries when disconnected.
Adding a named entry in the robot’s `cameras` mapping adds its uppercase topic family. A Triskel trial requires fresh registered depth from both cameras.

To support another physical RGB-D driver, define its settings as a subclass of
`HardwareCameraConfig` and add a branch in `hardware/cameras/factory.py` that
returns its `CameraSource`. The node uses the same publishing loop for each driver.

## SDK installation

`uv sync` installs the official `pyrealsense2` dependency on non-macOS platforms.
Linux also needs the [librealsense USB/udev setup](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md).
Use a USB 3 cable and port for the D435. `lsusb -t` should show `5000M`.
No RealSense package is installed on macOS; the physical driver intentionally fails
there. MuJoCo camera simulation and browser previews work without that SDK.

The project uses Python 3.12, for which the pinned official SDK provides Linux ARM64
wheels. Ubuntu on a 64-bit Raspberry Pi installs the SDK directly during `uv sync` or
deployment; no source build is needed. Python 3.13 is not supported by this project
because the pinned SDK has no matching ARM64 wheel.

## Image contract

Hardware and simulation publish the following camera streams.

| Topic | Pixels | Units / frame |
| --- | --- | --- |
| `CAMERA/<NAME>/RGB` | RGB-order `uint8`, H × W × 3 | RGB optical frame |
| `CAMERA/<NAME>/DEPTH` | `uint16`, H × W | `data * depth_scale` meters, depth optical frame |
| `CAMERA/<NAME>/REGISTERED_DEPTH` | `RegisteredDepth.image`, `uint16`, RGB H × W | Measured depth on the synchronized RGB pixel grid, RGB optical frame and intrinsics |

Zero depth is invalid. The D435 and D405 drivers align measured depth once per
frameset. The agent camera client selects this stream by default. Use the native
depth topic or `client.observe(depth_view="native")` for the depth sensor's own view.
Every image carries intrinsics, distortion coefficients, and its URDF optical frame ID.
Derive transforms to the robot base from the URDF. Pixels use fast lossless LZ4 transport;
16-bit samples use little-endian byte order on the wire.
Older zlib recordings remain readable. Update image consumers together: older versions
cannot decode LZ4 images.

`timestamp` is host monotonic receipt time shared by each pair. `device_timestamp` is
seconds in `timestamp_domain`; source clocks are not assumed synchronized to the host.
Subscribers keep the latest image per topic. Match host timestamps if a consumer needs
pairs, since independent receives may retrieve different framesets.

```python
from humanoid.types.camera import camera_topics
from humanoid.middleware.subscriber import Subscriber

topics = camera_topics("base")
subscriber = Subscriber([topics.rgb, topics.depth])
try:
    rgb = subscriber.receive(topics.rgb, timeout=1000)
    depth = subscriber.receive(topics.depth, timeout=1000)
    if depth is not None:
        depth_meters = depth.data.astype(float) * depth.depth_scale
finally:
    subscriber.close()
```

For receive buffer settings or subscribing to images from another host, see
[LCM networking](../../../../README.md#lcm-networking).

## Browser previews

The console requests the latest previews for every camera configured for the selected robot
and runtime in one request at up to 5 Hz, independently of control-status polling.
Each camera has a named RGB view and a colorized native-depth view with independent freshness.
RGB and colorized depth are JPEG previews; the original LCM images remain
unchanged. Amber means near, teal means middle distance, and blue means far. Values
outside the selected range saturate at the endpoints; invalid depth is black.

Previews disappear after two seconds without new messages and are labelled stale.
Freshness uses local LCM receipt time, so robot and console clocks need not match.
Preview defaults live in `CAMERA_PREVIEW_CONFIG`. Simulated depth is ideal rendered
geometry; it does not reproduce RealSense stereo noise or missing measurements.
