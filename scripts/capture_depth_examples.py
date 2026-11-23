"""Save RealSense RGB, depth, and available infrared images for depth-model comparisons."""

import argparse
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from humanoid.config.camera import REALSENSE_CAMERA_CONFIG


def calibration(frames, sensor, sdk):
    streams = {}
    to_rgb = {}
    for name, frame in frames.items():
        profile = frame.profile.as_video_stream_profile()
        intr = profile.get_intrinsics()
        streams[name] = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "distortion_model": str(intr.model),
            "distortion_coefficients": intr.coeffs,
            "frame_number": frame.get_frame_number(),
            "timestamp_ms": frame.get_timestamp(),
            "timestamp_domain": str(frame.get_frame_timestamp_domain()),
        }
        extr = profile.get_extrinsics_to(frames["rgb"].profile)
        # librealsense stores the source-to-target rotation in column-major order.
        to_rgb[name] = {
            "rotation": np.reshape(extr.rotation, (3, 3), order="F").tolist(),
            "translation_m": extr.translation,
        }
    return {
        "streams": streams,
        "optical_to_rgb": to_rgb,
        "depth_scale_m_per_unit": sensor.get_depth_scale(),
        "emitter_enabled_setting": sensor.get_option(sdk.option.emitter_enabled)
        if sensor.supports(sdk.option.emitter_enabled)
        else None,
    }


def save_example(folder, images, metadata):
    with TemporaryDirectory(dir=folder.parent) as staging:
        for name, pixels in images.items():
            Image.fromarray(pixels).save(Path(staging) / f"{name}.png")
        (Path(staging) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        Path(staging).rename(folder)


def configure_streams(sdk, serial, fps):
    config = REALSENSE_CAMERA_CONFIG
    serial = serial or config.serial_number
    context = sdk.context()
    devices = list(context.query_devices())
    if not serial:
        if len(devices) != 1:
            available = ", ".join(
                f"{d.get_info(sdk.camera_info.name)} ({d.get_info(sdk.camera_info.serial_number)})"
                for d in devices
            )
            raise SystemExit(f"Select a camera with --serial. Detected: {available or 'none'}")
        serial = devices[0].get_info(sdk.camera_info.serial_number)
    size = config.width, config.height

    def settings_for(indices):
        settings = sdk.config()
        settings.enable_device(serial)
        settings.enable_stream(sdk.stream.color, *size, sdk.format.rgb8, fps)
        settings.enable_stream(sdk.stream.depth, *size, sdk.format.z16, fps)
        for index in indices:
            settings.enable_stream(sdk.stream.infrared, index, *size, sdk.format.y8, fps)
        return settings

    pipeline = sdk.pipeline(context)
    settings = settings_for((1, 2))
    stereo = settings.can_resolve(sdk.pipeline_wrapper(pipeline))
    if not stereo:
        settings = settings_for((1,))
    return pipeline, settings, serial, stereo


def capture(output, sdk, serial=None, fps=None):
    config = REALSENSE_CAMERA_CONFIG
    fps = fps or config.fps
    pipeline, settings, serial, stereo = configure_streams(sdk, serial, fps)
    try:
        profile = pipeline.start(settings)
    except RuntimeError as error:
        raise SystemExit(
            f"Could not open streams on RealSense {serial}. Check USB and stop any active "
            f"camera session in the operator console. SDK error: {error}"
        ) from error
    try:
        device = profile.get_device()
        model = device.get_info(sdk.camera_info.name)
        print(f"{model} ({serial}), {config.width}x{config.height} at {fps} fps")
        if not stereo:
            print("Right IR unavailable in this profile. Saving RGB, depth and left IR.")
        sensor = device.first_depth_sensor()
        align = sdk.align(sdk.stream.color)
        session = output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        session.mkdir(parents=True)
        print(f"Saving to {session.resolve()}")
        for _ in range(fps):  # Let auto-exposure settle for about one second.
            pipeline.wait_for_frames()
        index = 1
        while True:
            note = input(f"Scene {index}: enter a note or press Enter to capture (q to finish): ")
            if note.strip().lower() == "q":
                break
            pipeline.wait_for_frames()  # Discard the frameset queued while the operator was typing.
            bundle = pipeline.wait_for_frames()
            frames = {
                "rgb": bundle.get_color_frame(),
                "depth": bundle.get_depth_frame(),
                "ir_left": bundle.get_infrared_frame(1),
            }
            if stereo:
                frames["ir_right"] = bundle.get_infrared_frame(2)
            if not all(frames.values()):
                print("Incomplete frameset. Press Enter to retry this scene.")
                continue
            images = {name: np.asanyarray(frame.get_data()) for name, frame in frames.items()}
            aligned = align.process(bundle).get_depth_frame()
            images["depth_in_rgb"] = np.asanyarray(aligned.get_data())
            coverage = np.count_nonzero(images["depth"]) / images["depth"].size
            metadata = {
                "note": note,
                "camera_model": model,
                "fps": fps,
                "serial_number": device.get_info(sdk.camera_info.serial_number),
                "native_valid_fraction": coverage,
                **calibration(frames, sensor, sdk),
            }
            folder = session / f"scene-{index:03d}"
            save_example(folder, images, metadata)
            print(f"Saved {folder.name} ({coverage:.1%} valid native depth)")
            index += 1
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/depth-examples"))
    parser.add_argument("--serial", help="RealSense serial number (required with multiple cameras)")
    parser.add_argument("--fps", type=int, choices=(5, 15, 30), help="Capture frame rate")
    args = parser.parse_args()
    try:
        sdk = importlib.import_module("pyrealsense2")
    except ImportError as error:
        raise SystemExit("Run this on Triskel with the project's RealSense environment.") from error
    try:
        capture(args.output, sdk, args.serial, args.fps)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
