"""Run RGB/depth approach-and-squeeze trials: uv run python -m humanoid.simulation.cube_pick.

Cube and ground are fixed visual geometry without contacts. Robot motion uses
MuJoCo actuators and the normal operational-space controller. Ground truth is
used only to construct the scene and score the final pose, never by the policy.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from humanoid.config.cube_pick import (
    CUBE_PICK_CONFIG,
    CUBE_PICK_CONTROLLER_CONFIG,
    CUBE_PICK_SUBSTEPS,
    CUBE_PICK_TEST_POSES,
    WOOD_CUBE_PICK_CONFIG,
    WOOD_CUBE_TEST_POSES,
)
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.policy.cube_pick import CubePickPolicy
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.cube_pick import CubeDetectionMethod, CubePickConfig, CubePickPhase
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotJointCommand

MAX_GRIP_ERROR_METERS = 0.01
MAX_CLOSURE_ERROR_RADIANS = 0.05


def configure_cube_scene(spec: mujoco.MjSpec, config: CubePickConfig = CUBE_PICK_CONFIG) -> None:
    """Add a lit floor and one repositionable, contact-free cube."""
    spec.worldbody.add_geom(
        name="cube_floor",
        pos=[0, 0, config.ground_height],
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.01],
        rgba=[0.25, 0.25, 0.25, 1],
        contype=0,
        conaffinity=0,
    )
    cube = spec.worldbody.add_body(name="pick_cube_body", mocap=True)
    cube.add_geom(
        name="pick_cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[config.cube_size / 2] * 3,
        rgba=[0.65, 0.43, 0.23, 1]
        if config.detection_method is CubeDetectionMethod.DEPTH_SIDES
        else [0.9, 0.03, 0.03, 1],
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        pos=[0, 0.1, 1], dir=[0, 0, -1], ambient=[0.5, 0.5, 0.5], castshadow=False
    )


def run_trials(
    output: Path,
    config: CubePickConfig = CUBE_PICK_CONFIG,
    poses: tuple[tuple[float, float, float], ...] = CUBE_PICK_TEST_POSES,
) -> list[dict[str, object]]:
    """Render and execute every configured pose; save images and measured errors."""
    output.mkdir(parents=True, exist_ok=True)
    engine = NativeMujocoEngine(
        TRISKEL_CONFIG, configure_scene=partial(configure_cube_scene, config=config)
    )
    reports: list[dict[str, object]] = []
    try:
        for index, (x, y, yaw) in enumerate(poses):
            engine.reset()
            engine.data.mocap_pos[0] = [x, y, config.ground_height + config.cube_size / 2]
            engine.data.mocap_quat[0] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(engine.model, engine.data)
            frames = engine.read_camera_frames(0.0)
            assert frames is not None
            Image.fromarray(frames.rgb.data).save(output / f"{index}_rgb.png")
            Image.fromarray(frames.depth.data).save(output / f"{index}_depth.png")
            policy = CubePickPolicy(engine.robot, config)
            controller = OperationalSpaceController(engine.robot, CUBE_PICK_CONTROLLER_CONFIG)
            phases: list[str] = []
            while engine.data.time < config.phase_timeout * 4:
                state = engine.read_robot_state(float(engine.data.time))
                action = policy(Observation(robot_state=state, camera_frames=frames))
                if not phases or phases[-1] != policy.phase.value:
                    phases.append(policy.phase.value)
                if policy.phase is CubePickPhase.FAILED:
                    break
                if action.tool_pose is not None:
                    # Match the normal controller's integrated command reference.
                    # Policy phase transitions still use measured MuJoCo state.
                    if controller.configuration is None:
                        controller.update_state(state.joint_positions)
                    result = controller.compute_control(action.tool_pose)
                    if action.gripper_positions is not None:
                        engine.robot.set_gripper_positions(result.q, action.gripper_positions)
                    engine.apply_joint_command(
                        RobotJointCommand(
                            timestamp=state.timestamp,
                            joint_positions=result.q,
                        )
                    )
                engine.step(CUBE_PICK_SUBSTEPS)
                if policy.is_done:
                    break
            q = engine.read_robot_state(float(engine.data.time)).joint_positions
            tool = engine.robot.get_tool_command_pose(q)
            grip = tool.translation + tool.rotation @ np.asarray(config.grip_offset)
            true_center = np.array([x, y, config.ground_height + config.cube_size / 2])
            error = float(np.linalg.norm(grip - true_center))
            gripper = q[engine.robot.get_gripper_position_indices()]
            closed_error = float(np.max(np.abs(gripper - config.closed_position)))
            detection = policy.detection
            report: dict[str, object] = {
                "pose": [x, y, float(yaw)],
                "phase": policy.phase.value,
                "phases": phases,
                "elapsed_sim_seconds": float(engine.data.time),
                "detected_points": 0 if detection is None else detection.point_count,
                "perception_error_m": None
                if detection is None
                else float(np.linalg.norm(detection.center - true_center)),
                "grip_center_error_m": error,
                "gripper_error_rad": closed_error,
                "success": policy.is_done
                and error < MAX_GRIP_ERROR_METERS
                and closed_error < MAX_CLOSURE_ERROR_RADIANS,
                "failure_reason": policy.failure_reason,
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
            _save_overview(engine, output / f"{index}_final.png")
    finally:
        engine.close()
    (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def _save_overview(engine: NativeMujocoEngine, path: Path) -> None:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0.18, 0.13]
    camera.distance = 0.65
    camera.azimuth = 135
    camera.elevation = -25
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    with mujoco.Renderer(engine.model, height=480, width=640) as renderer:
        renderer.update_scene(engine.data, camera=camera, scene_option=options)
        Image.fromarray(renderer.render()).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/cube_pick"))
    parser.add_argument(
        "--wood", action="store_true", help="Use the 45 mm depth-based wood-cube profile"
    )
    args = parser.parse_args()
    reports = (
        run_trials(args.output, WOOD_CUBE_PICK_CONFIG, WOOD_CUBE_TEST_POSES)
        if args.wood
        else run_trials(args.output)
    )
    if not all(report["success"] for report in reports):
        raise SystemExit("One or more approach-and-squeeze trials failed; see results.json.")


if __name__ == "__main__":
    main()
