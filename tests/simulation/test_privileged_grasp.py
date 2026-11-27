"""Floor pickup using ground-truth toy poses and the native robot actuators."""

import csv
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.config.robot.panda import PANDA_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.config.teleop import get_oculus_teleop_policy_config
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.simulation import SimulationScene

GRASP_OUTPUT_ENVIRONMENT_VARIABLE = "HUMANOID_GRASP_OUTPUT"

MIN_LIFT_M = 0.060
HOLD_SECONDS = 10.0
MAX_HOLD_HEIGHT_RANGE_M = 0.002
MAX_CUBE_OVERLAP_M = 0.001
MAX_CUBE_PRELIFT_RISE_M = 0.001
MAX_CUBE_FLOOR_OVERLAP_M = 0.0003
MAX_CUBE_CARRY_SLIP_M = 0.0005
MIN_CUBE_PUSH_M = 0.04
MAX_RELEASE_HEIGHT_M = 0.02
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
        robot = self.engine.robot
        names = [robot.joint_idx_to_name(idx) for idx in robot.get_arm_joint_indices()]
        self.dofs = [model.joint(name).dofadr[0] for name in names]
        self.actuators = [model.actuator(name).id for name in names]
        self.limits = model.actuator_ctrlrange[self.actuators].T
        assert config.gripper is not None
        (gripper_name,) = config.gripper.joint_names
        self.gripper = model.actuator(gripper_name).id
        lower, upper = model.actuator_ctrlrange[self.gripper]
        self.closed_limit = upper if config.gripper.closes_at_upper_limit else lower
        fingers = (
            ("finger_1_link", "finger_2_link")
            if self.triskel
            else ("panda_leftfinger", "panda_rightfinger")
        )
        self.finger_slots = {model.body(name).id: slot for slot, name in enumerate(fingers)}

    def move(
        self,
        rotation_target: Rotation,
        target: np.ndarray,
        seconds: float,
        sample: Callable[[], None] = lambda: None,
    ) -> None:
        """Servo the measured pinch frame with a damped arm-only Jacobian."""
        model, data = self.engine.model, self.engine.data
        marker = data.site("pinch")
        jac = np.empty((6, model.nv))
        dt = self.engine.physics_timestep
        for _ in range(round(seconds / dt)):
            rotation = Rotation.from_matrix(marker.xmat.reshape(3, 3))
            error = np.r_[target - marker.xpos, (rotation_target * rotation.inv()).as_rotvec()]
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
            slot = self.finger_slots.get(other)
            if slot is None:
                slot = 2
            else:
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


@pytest.fixture(scope="module")
def triskel_rig() -> Iterator[GraspRig]:
    rig = GraspRig(TRISKEL_CONFIG)
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


@dataclass(frozen=True)
class PinchApproach:
    travel: float | None
    offset_m: float
    yaw_deg: int
    tilt_deg: int

    def rotation(self, triskel: bool) -> Rotation:
        top_down = Rotation.from_euler("x", -90 if triskel else 180, degrees=True)
        return Rotation.from_euler("zy", [self.yaw_deg, self.tilt_deg], degrees=True) * top_down

    def close_commands(self, rig: GraspRig, width: float) -> tuple[np.ndarray, float]:
        opened = rig.engine.data.ctrl[rig.gripper]
        if self.travel is None:
            # Preserve the original width-matched benchmark.
            closed = (0.050 - width + 0.002) / 0.023 if rig.triskel else width / 2 - 0.004
            return np.linspace(opened, closed, 50), 0.015

        closed = opened + self.travel * (rig.closed_limit - opened)
        teleop = get_oculus_teleop_policy_config()
        ticks = round(self.travel * teleop.gripper_close_time / teleop.dt)
        return np.linspace(opened, closed, ticks + 1)[1:], teleop.dt


def prepare_pickup(rig: GraspRig, kind: str, approach: PinchApproach) -> np.ndarray:
    engine = rig.engine
    engine.reset()
    model, data = engine.model, engine.data
    address = model.jnt_qposadr[model.body(f"{kind}_0").jntadr[0]]
    # Place the same free toy within reach, with its narrow side across the jaws.
    data.qpos[address : address + 2] = (0, 0.30) if rig.triskel else (0.45, 0)
    if rig.triskel and kind == "car":
        data.qpos[address + 3 : address + 7] = [2**-0.5, 0, 0, 2**-0.5]
    engine.step(round(0.5 / engine.physics_timestep))
    target = data.body(f"{kind}_0").xpos.copy()
    if kind == "car":
        # Aim Panda's short pads at the cabin, above the free-spinning wheels.
        target[2] += 0.011 if not rig.triskel and approach.travel is not None else 0.003
    target[0 if rig.triskel else 1] += approach.offset_m
    return target


def assert_released(rig: GraspRig, name: str, start_z: float) -> None:
    toy = rig.engine.data.body(name)
    loads, _ = rig.contact_metrics(toy.id)
    assert np.all(loads[:2] == 0) and loads[2] > 0
    assert toy.xpos[2] - start_z < MAX_RELEASE_HEIGHT_M


def cube_floor_overlap(engine: NativeMujocoEngine) -> float:
    """Depth of the lowest rotated box corner below the room floor."""
    cube_geom = engine.model.geom(int(engine.model.body("cube_0").geomadr[0]))
    cube = engine.data.geom(cube_geom.id)
    half_height = np.abs(cube.xmat.reshape(3, 3)[2]) @ cube_geom.size
    return engine.data.geom("task_floor").xpos[2] - (cube.xpos[2] - half_height)


def test_cube_push_on_floor(triskel_rig: GraspRig):
    rig = triskel_rig
    approach = PinchApproach(None, 0, 0, 7)
    target = prepare_pickup(rig, "cube", approach)
    move = partial(rig.move, approach.rotation(rig.triskel))
    start = target + np.array([-0.075, 0, -0.005])
    move(start + np.array([0, 0, 0.07]), 2.5)
    move(start, 1.5)
    depths = []
    for x in np.linspace(0, 0.08, 80):
        move(
            start + np.array([x, 0, 0]), 0.02, lambda: depths.append(cube_floor_overlap(rig.engine))
        )

    assert rig.engine.data.body("cube_0").xpos[0] - target[0] > MIN_CUBE_PUSH_M
    assert max(depths) < MAX_CUBE_FLOOR_OVERLAP_M


def test_cube_stays_in_skewed_grasp_during_arm_reversals(triskel_rig: GraspRig):
    rig = triskel_rig
    approach = PinchApproach(1.0, -0.004, -12, -7)
    target = prepare_pickup(rig, "cube", approach)
    above = target + np.array([0, 0, 0.08])
    rotation = approach.rotation(rig.triskel)
    move = partial(rig.move, rotation)
    move(above, 2.5)
    move(target, 1.5)
    commands, interval = approach.close_commands(rig, 0.036)
    depths = []
    for command in commands:
        rig.engine.data.ctrl[rig.gripper] = command
        move(target, interval, lambda: depths.append(cube_floor_overlap(rig.engine)))
    move(target, 0.7, lambda: depths.append(cube_floor_overlap(rig.engine)))
    assert max(depths) < MAX_CUBE_FLOOR_OVERLAP_M

    for z in np.linspace(0, 0.08, 80):
        move(target + np.array([0, 0, z]), 0.02)
    move(above, 0.3)
    cube = rig.engine.data.body("cube_0")
    pinch = rig.engine.data.site("pinch")

    def cube_in_hand() -> np.ndarray:
        return pinch.xmat.reshape(3, 3).T @ (cube.xpos - pinch.xpos)

    initial = cube_in_hand()

    def check_carry() -> None:
        loads, _ = rig.contact_metrics(cube.id)
        assert np.all(loads[:2] > 0) and loads[2] == 0
        assert np.linalg.norm(cube_in_hand() - initial) < MAX_CUBE_CARRY_SLIP_M

    for dx, dy, yaw in [(0.05, -0.025, 18), (-0.05, 0, -18), (0, 0, 0)]:
        rig.move(
            Rotation.from_euler("z", yaw, degrees=True) * rotation,
            above + np.array([dx, dy, 0]),
            1.1,
            check_carry,
        )


@pytest.mark.parametrize(
    ("kind", "width"), [("cube", 0.036), ("cylinder", 0.034), ("ball", 0.042), ("car", 0.022)]
)
@pytest.mark.parametrize(
    "approach",
    [
        PinchApproach(None, 0, 0, 0),
        PinchApproach(0.80, -0.003, -8, -4),
        PinchApproach(1.0, 0.003, 8, 4),
    ],
    ids=["aligned", "partial-close", "full-close"],
)
def test_privileged_floor_pickup(
    rig: GraspRig, kind: str, width: float, approach: PinchApproach
) -> None:
    target = prepare_pickup(rig, kind, approach)
    data = rig.engine.data
    name = f"{kind}_0"
    toy = data.body(name)
    start_z = toy.xpos[2]
    above = target + np.array([0, 0, 0.08])
    move = partial(rig.move, approach.rotation(rig.triskel))
    opened = data.ctrl[rig.gripper]
    commands, interval = approach.close_commands(rig, width)

    with ExitStack() as stack:
        output = os.getenv(GRASP_OUTPUT_ENVIRONMENT_VARIABLE)
        directory = "" if approach.travel is None else f"close-{approach.travel:.0%}"
        save_frame = (
            record_trial(stack, rig, name, Path(output) / directory)
            if output
            else lambda phase: None
        )
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

        move(above, 2.5, lambda: record("approach"))
        move(target, 1.5, lambda: record("descend"))
        for command in commands:
            data.ctrl[rig.gripper] = command
            move(target, interval, lambda: record("close"))
        move(target, 0.7, lambda: record("squeeze"))
        for z in np.linspace(0, 0.08, 80):
            move(target + np.array([0, 0, z]), 0.02, lambda: record("lift"))
        move(above, 0.5, lambda: record("settle"))
        loads: list[np.ndarray] = []
        heights: list[float] = []

        def sample_hold() -> None:
            record("hold")
            heights.append(toy.xpos[2] - start_z)
            loads.append(rig.contact_metrics(toy.id)[0])

        move(above, HOLD_SECONDS, sample_hold)
        data.ctrl[rig.gripper] = opened
        move(above, 1.0, lambda: save_frame("release"))
        assert_released(rig, name, start_z)

    assert min(heights) > MIN_LIFT_M, f"Minimum held lift was {min(heights):.4f} m"
    height_range = max(heights) - min(heights)
    assert height_range < MAX_HOLD_HEIGHT_RANGE_M, f"Hold height varied by {height_range:.4f} m"
    assert np.all(np.array(loads)[:, :2] > 0), "Lost loaded bilateral finger contact"
    assert np.all(np.array(loads)[:, 2] == 0), "Toy contacted another surface during hold"
    if check_cube:
        assert max_depth < MAX_CUBE_OVERLAP_M, f"Finger penetration reached {max_depth:.4f} m"
        assert prelift_rise < MAX_CUBE_PRELIFT_RISE_M, f"Cube rose {prelift_rise:.4f} m before lift"


@pytest.mark.parametrize(
    ("approach", "height_offset"),
    [
        (PinchApproach(1.0, 0.003, 8, 4), 0.002),
        (PinchApproach(1.0, -0.004, -12, -7), 0.003),
        (PinchApproach(1.0, 0.004, 15, 8), 0.006),
    ],
)
def test_triskel_carries_car_with_off_center_full_close(
    approach: PinchApproach, height_offset: float
) -> None:
    rig = GraspRig(TRISKEL_CONFIG)
    try:
        target = prepare_pickup(rig, "car", approach)
        target[2] += height_offset
        car = rig.engine.data.body("car_0")
        start_z = car.xpos[2]
        move = partial(rig.move, approach.rotation(rig.triskel))

        move(target + np.array([0, 0, 0.08]), 2.5)
        move(target, 1.5)
        opened = rig.engine.data.ctrl[rig.gripper]
        for command in np.linspace(opened, rig.closed_limit, 30):
            rig.engine.data.ctrl[rig.gripper] = command
            move(target, 1 / 30)
        move(target, 0.2)
        carry = np.array([0.025, 0, 0.08])
        for fraction in np.linspace(0, 1, 40):
            move(target + fraction * carry, 0.03)

        def check_hold() -> None:
            assert car.xpos[2] - start_z > MIN_LIFT_M
            loads, _ = rig.contact_metrics(car.id)
            assert np.all(loads[:2] > 0) and loads[2] == 0

        move(target + carry, 3, check_hold)
    finally:
        rig.engine.close()
