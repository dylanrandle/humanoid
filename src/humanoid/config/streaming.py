"""WebRTC media limits for local-network camera streaming."""

from humanoid.types.streaming import StreamingConfig


def get_streaming_config() -> StreamingConfig:
    return StreamingConfig(
        fps=15,
        max_width=640,
        max_sessions=16,
        negotiation_timeout=15,
        idle_timeout=30,
    )
