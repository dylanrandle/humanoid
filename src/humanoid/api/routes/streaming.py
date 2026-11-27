"""WebRTC signaling on the same HTTP listener as robot control."""

from dataclasses import asdict

from flask import Flask, jsonify, request

from humanoid.api.application import BackendApplication
from humanoid.config.camera import CAMERA_DISPLAY_CONFIG
from humanoid.types.api import ApiRoute
from humanoid.types.streaming import StreamOffer


def register_streaming_routes(app: Flask, backend: BackendApplication) -> None:
    @app.get(ApiRoute.CAMERAS)
    def cameras():
        return jsonify(
            cameras=list(backend.camera_frame_services),
            generation=backend.camera_generation,
            depth_near=CAMERA_DISPLAY_CONFIG.depth_near_meters,
            depth_ranges=CAMERA_DISPLAY_CONFIG.depth_ranges_meters,
            streams=backend.stream_telemetry.snapshot(),
        )

    @app.post(ApiRoute.STREAMS)
    def offer_stream():
        offer = StreamOffer.model_validate(request.get_json())
        source = backend.camera_frame_services.get(offer.camera)
        if source is None:
            raise ValueError("Unknown camera.")
        answer = backend.streaming_service.offer(offer, lambda: source.read_frame(offer.stream))
        return jsonify(asdict(answer))

    @app.delete(ApiRoute.STREAM)
    def close_stream(session_id: str):
        backend.streaming_service.remove(session_id)
        return jsonify(ok=True)
