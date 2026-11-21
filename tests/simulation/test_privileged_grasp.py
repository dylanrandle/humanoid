"""Floor pickup using ground-truth toy poses and the native robot actuators."""

import csv
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.config.robot.panda import PANDA_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.simulation import SimulationScene

MIN_LIFT_M = 0.060
HOLD_SECONDS = 10.0
MAX_HOLD_HEIGHT_RANGE_M = 0.002
MAX_CUBE_OVERLAP_M = 0.001
MAX_CUBE_PRELIFT_RISE_M = 0.001
FPS = 25


class GraspRig:
    def __init__(self, config: RobotConfig) -> None:
        self.triskel = config.name == RobotName.TRISKEL
        frame = "gripper_base_link" if self.triskel else "panda_hand"
        offset = [0.002, 0.127, 0] if self.triskel else [0, 0, 0.104]
        simulation = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=SimulationScene.TOY_CLEANUP)

        def add_marker(spec: mujoco.MjSpec) -> None:
            spec.body(frame).add_site(name="pinch", pos=offset, size=[0.001], group=3)

        self.engine = NativeMujocoEngine(config, simulation, configure_scene=add_marker)
        model = self.engine.model
        names = [f"arm_{i}" if self.triskel else f"panda_joint{i}" for i in range(1, 8)]
        self.dofs = [model.joint(name).dofadr[0] for name in names]
        self.actuators = [model.actuator(name).id for name in names]
        self.limits = model.actuator_ctrlrange[self.actuators].T
        self.rotation = Rotation.from_euler("x", -90 if self.triskel else 180, degrees=True)
        self.gripper = model.actuator("gripper_1" if self.triskel else "panda_finger_joint1").id
        fingers = (
            ("finger_1_link", "finger_2_link")
            if self.triskel
            else ("panda_leftfinger", "panda_rightfinger")
        )
        self.fingers = [model.body(name).id for name in fingers]

    def move(self, target: np.ndarray, seconds: float, sample: Callable[[], None]) -> None:
        """Servo the measured pinch frame with a damped arm-only Jacobian."""
        model, data = self.engine.model, self.engine.data
        marker = data.site("pinch")
        jac = np.empty((6, model.nv))
        dt = self.engine.physics_timestep
        for _ in range(round(seconds / dt)):
            rotation = Rotation.from_matrix(marker.xmat.reshape(3, 3))
            error = np.r_[target - marker.xpos, (self.rotation * rotation.inv()).as_rotvec()]
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], marker.id)
            arm = jac[:, self.dofs]
            delta = arm.T @ np.linalg.solve(arm @ arm.T + 1e-4 * np.eye(6), error)
            step = np.clip(delta * 5, -1, 1) * dt
            data.ctrl[self.actuators] = np.clip(data.ctrl[self.actuators] + step, *self.limits)
            self.engine.step()
            sample()

    def contact_metrics(self, toy_id: int) -> tuple[np.ndarray, float]:
        """Finger and support loads (N), and deepest finger-to-toy overlap (m)."""
        model, data = self.engine.model, self.engine.data
        forces = np.zeros(3)
        penetration = 0.0
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact):
            first, second = model.geom_bodyid[contact.geom]
            if model.body_rootid[first] == toy_id:
                other = second
            elif model.body_rootid[second] == toy_id:
                other = first
            else:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            slot = 2
            if other in self.fingers:
                slot = self.fingers.index(other)
                penetration = max(penetration, -contact.dist)
            forces[slot] += wrench[0]
        return forces, penetration


@pytest.fixture(scope="module", params=[PANDA_CONFIG, TRISKEL_CONFIG], ids=["panda", "triskel"])
def rig(request: pytest.FixtureRequest) -> Iterator[GraspRig]:
    rig = GraspRig(request.param)
    try:
        yield rig
    finally:
        rig.engine.close()


def record_trial(stack: ExitStack, rig: GraspRig, name: str, path: Path) -> Callable[[str], None]:
    engine = rig.engine
    toy = engine.data.body(name)
    start_z = toy.xpos[2]
    path.mkdir(parents=True, exist_ok=True)
    stem = path / f"{engine.robot.config.name}-{name}"
    trace = csv.writer(stack.enter_context(stem.with_suffix(".csv").open("w", newline="")))
    trace.writerow(
        [
            "time_s",
            "phase",
            "lift_m",
            "toy_x_m",
            "toy_y_m",
            "toy_z_m",
            "pinch_x_m",
            "pinch_y_m",
            "pinch_z_m",
            "gripper_command",
            "finger_1_N",
            "finger_2_N",
            "other_contact_N",
            "finger_penetration_m",
        ]
    )
    renderer = stack.enter_context(mujoco.Renderer(engine.model, height=360, width=480))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = toy.xpos + np.array([0, 0, 0.075])
    camera.distance = 0.46
    camera.azimuth = 62 if rig.triskel else -38
    camera.elevation = -18
    options = mujoco.MjvOption()
    options.geomgroup[3] = options.sitegroup[3] = 0
    writer = cv2.VideoWriter(
        str(stem.with_suffix(".mp4")), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (480, 360)
    )
    stack.callback(writer.release)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {stem}")
    next_frame = engine.data.time

    def record(phase: str) -> None:
        nonlocal next_frame
        data = engine.data
        if data.time + 1e-9 < next_frame:
            return
        next_frame += 1 / FPS
        lift = toy.xpos[2] - start_z
        forces, penetration = rig.contact_metrics(toy.id)
        trace.writerow(
            [
                data.time,
                phase,
                lift,
                *toy.xpos,
                *data.site("pinch").xpos,
                data.ctrl[rig.gripper],
                *forces,
                penetration,
            ]
        )
        renderer.update_scene(data, camera=camera, scene_option=options)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            f"{stem.name}  {phase}  {lift * 1000:.0f} mm",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    return record


@pytest.mark.parametrize(
    ("kind", "width"), [("cube", 0.036), ("cylinder", 0.034), ("ball", 0.042), ("car", 0.022)]
)
def test_privileged_floor_pickup(rig: GraspRig, kind: str, width: float) -> None:
    engine = rig.engine
    engine.reset()
    model, data = engine.model, engine.data
    name = f"{kind}_0"
    toy = data.body(name)
    address = model.jnt_qposadr[model.body(name).jntadr[0]]
    # Place the same free toy within each arm's reach, with its narrow side across the jaws.
    data.qpos[address : address + 2] = (0, 0.30) if rig.triskel else (0.45, 0)
    if rig.triskel and kind == "car":
        data.qpos[address + 3 : address + 7] = [2**-0.5, 0, 0, 2**-0.5]
    engine.step(100)
    target = toy.xpos.copy()
    start_z = target[2]
    target[2] += 0.003 if kind == "car" else 0
    above = target + np.array([0, 0, 0.08])
    # Triskel's rack moves 11.5 mm per radian per jaw. Panda commands one jaw in metres.
    closed = (0.050 - width + 0.002) / 0.023 if rig.triskel else width / 2 - 0.004

    with ExitStack() as stack:
        output = os.getenv("HUMANOID_GRASP_OUTPUT")
        save_frame = record_trial(stack, rig, name, Path(output)) if output else lambda phase: None
        check_cube = rig.triskel and kind == "cube"
        max_depth = 0.0
        prelift_rise = 0.0

        def record(phase: str) -> None:
            nonlocal max_depth, prelift_rise
            save_frame(phase)
            if check_cube:
                max_depth = max(max_depth, rig.contact_metrics(toy.id)[1])
                if phase in ("approach", "descend", "close", "squeeze"):
                    prelift_rise = max(prelift_rise, toy.xpos[2] - start_z)

        rig.move(above, 2.5, lambda: record("approach"))
        rig.move(target, 1.5, lambda: record("descend"))
        for command in np.linspace(data.ctrl[rig.gripper], closed, 50):
            data.ctrl[rig.gripper] = command
            rig.move(target, 0.015, lambda: record("close"))
        rig.move(target, 0.7, lambda: record("squeeze"))
        for z in np.linspace(0, 0.08, 80):
            rig.move(target + np.array([0, 0, z]), 0.02, lambda: record("lift"))
        rig.move(above, 0.5, lambda: record("settle"))
        loads: list[np.ndarray] = []
        heights: list[float] = []

        def sample_hold() -> None:
            record("hold")
            heights.append(toy.xpos[2] - start_z)
            loads.append(rig.contact_metrics(toy.id)[0])

        rig.move(above, HOLD_SECONDS, sample_hold)

    assert min(heights) > MIN_LIFT_M, f"Minimum held lift was {min(heights):.4f} m"
    height_range = max(heights) - min(heights)
    assert height_range < MAX_HOLD_HEIGHT_RANGE_M, f"Hold height varied by {height_range:.4f} m"
    assert np.all(np.array(loads)[:, :2] > 0), "Lost loaded bilateral finger contact"
    assert np.all(np.array(loads)[:, 2] == 0), "Toy contacted another surface during hold"
    if check_cube:
        assert max_depth < MAX_CUBE_OVERLAP_M, f"Finger penetration reached {max_depth:.4f} m"
        assert prelift_rise < MAX_CUBE_PRELIFT_RISE_M, f"Cube rose {prelift_rise:.4f} m before lift"
