"""HTTP routes for robot state, camera captures, and motion control."""

from dataclasses import asdict
from http import HTTPStatus

from flask import Flask, jsonify, request

from humanoid.api.application import BackendApplication
from humanoid.api.serialization import encode_frames, encode_state
from humanoid.types.api import ApiRoute
from humanoid.types.camera import DepthView
from humanoid.types.motion import MotionRequest, MotionSession, MotionSubmission


def register_robot_routes(app: Flask, backend: BackendApplication) -> None:
    @app.get(ApiRoute.STATE)
    def robot_state():
        runtime_id, observation = backend.observe()
        return jsonify(
            runtime_id=runtime_id,
            motion_session_id=backend.motion_service.motion_session_id,
            motion_ready=backend.motion_service.ready,
            state=encode_state(observation.robot_state),
            mode=observation.mode,
        )

    @app.get(ApiRoute.CAPTURE)
    def capture():
        runtime_id, captures = backend.capture()
        return jsonify(
            runtime_id=runtime_id,
            views={
                name: {
                    "state": encode_state(obs.robot_state),
                    "mode": obs.mode,
                    "camera": encode_frames(frames, depth_view=DepthView.BOTH),
                }
                for name, (obs, frames) in captures.items()
            },
        )

    @app.get(ApiRoute.CAMERA)
    def camera():
        if set(request.args) - {"depth_view"} or len(request.args.getlist("depth_view")) > 1:
            raise ValueError("Expected one depth_view camera selector.")
        view = DepthView(request.args.get("depth_view", DepthView.ALIGNED))
        frames = backend.observation_service.camera_frames(depth_view=view)
        return jsonify(encode_frames(frames, depth_view=view))

    @app.post(ApiRoute.MOTIONS)
    def submit_motion():
        submission = MotionSubmission.model_validate(request.get_json())
        status = backend.motion_service.submit(
            MotionRequest(
                submission.motion_id,
                submission.operation,
                submission.arguments,
                submission.expires_in_seconds,
            ),
            submission.motion_session_id,
        )
        return jsonify(asdict(status)), HTTPStatus.ACCEPTED

    @app.get(ApiRoute.MOTION)
    def motion_status(motion_id: str):
        session = MotionSession.model_validate(request.args.to_dict())
        return jsonify(asdict(backend.motion_service.status(motion_id, session.motion_session_id)))

    @app.post(ApiRoute.MOTION_CANCEL)
    def cancel_motion(motion_id: str):
        session = MotionSession.model_validate(request.get_json())
        return jsonify(asdict(backend.motion_service.cancel(motion_id, session.motion_session_id)))

    @app.post(ApiRoute.RELEASE_AUTONOMY)
    def release_autonomy():
        session = MotionSession.model_validate(request.get_json())
        backend.release_autonomy(session.motion_session_id)
        return jsonify({}), HTTPStatus.ACCEPTED
