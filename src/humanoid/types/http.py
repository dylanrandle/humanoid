"""HTTP contract shared by the agent and operator console."""

from enum import StrEnum


class HttpKey(StrEnum):
    AFTER = "after"
    AGE_SECONDS = "age_seconds"
    CAMERA_MODEL = "camera_model"
    DEPTH = "depth"
    DEPTH_FAR = "depth_far"
    DEPTH_SCALE = "depth_scale"
    DEPTH_VIEW = "depth_view"
    DEVICE_TIMESTAMP = "device_timestamp"
    DISTORTION_COEFFICIENTS = "distortion_coefficients"
    ENCODING = "encoding"
    FRAME_ID = "frame_id"
    FRAME_NUMBER = "frame_number"
    INTRINSICS = "intrinsics"
    PNG_BASE64 = "png_base64"
    REGISTERED = "registered"
    RGB = "rgb"
    RGB_FRAME_NUMBER = "rgb_frame_number"
    SERIAL_NUMBER = "serial_number"
    TIMESTAMP = "timestamp"
    TIMESTAMP_DOMAIN = "timestamp_domain"
