# Cameras

Hardware and simulation expose named RGB-D cameras. Triskel uses a bottom-plate
D435 and a wrist D405, selected by serial number in its robot configuration.

## Hardware setup

`uv sync` installs the official `pyrealsense2` SDK on non-macOS platforms.
Linux also needs the [librealsense USB permissions setup](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md).
Use a USB 3 cable and port for the D435; `lsusb -t` should show `5000M`.
The project's Python 3.12 dependency supports Linux ARM64 wheels, including a
64-bit Raspberry Pi. Physical RealSense cameras are not supported on macOS;
use simulation there.

## Understanding depth

Each camera publishes RGB, native depth, and measured depth registered to RGB.
Native depth uses the depth sensor's optical frame. Registered depth shares the
synchronized RGB pixel grid and calibration, making it suitable for selecting
points in a color image.

Multiply a depth sample by `depth_scale` to obtain axial metres. Zero means no
measurement. `RobotClient.observe()` selects registered depth by default and
returns `None` for depth when a fresh synchronized pair is unavailable. Use
`depth_view="native"` for the sensor's native view.

Simulation provides ideal geometry rather than RealSense noise or missing samples.
Browser video is a display preview; use the client for metric depth and calibration.
See [Agent control](../agents.md) for client usage and optional perception tools.

## Configuration and integration

The robot's `cameras` mapping defines names, sensor settings, and URDF optical
frames. Shared defaults live in [camera configuration](../../src/humanoid/config/camera.py).
To add a physical driver, implement `CameraSource` and register its typed settings
in the [camera factory](../../src/humanoid/hardware/cameras/factory.py).
Message fields, timestamp semantics, and topic naming live in
[camera types](../../src/humanoid/types/camera.py).

## See also

- [Documentation index](../README.md)
- [Hardware](README.md)
- [Networking and video transport](../networking.md)
- [Simulation cameras](../simulation.md#cameras)
