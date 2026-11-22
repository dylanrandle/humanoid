import pytest

from humanoid.config.robot import TRISKEL_CONFIG
from humanoid.hardware.cameras.factory import create_camera
from humanoid.hardware.cameras.realsense import RealSenseCamera
from humanoid.types.camera import HardwareCameraConfig


@pytest.mark.parametrize("name", ["base", "wrist"])
def test_creates_driver_with_configured_mount(name):
    config = TRISKEL_CONFIG.cameras[name].hardware
    assert config is not None
    camera = create_camera(config)
    assert isinstance(camera, RealSenseCamera)
    assert camera.config is config


def test_unknown_hardware_requires_a_driver():
    config = HardwareCameraConfig(
        width=640, height=480, fps=30, rgb_frame="rgb", depth_frame="depth"
    )
    with pytest.raises(TypeError, match="Unsupported camera hardware config: HardwareCameraConfig"):
        create_camera(config)
