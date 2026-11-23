"""Carry a single cube into the bin using ground-truth geometry and native actuators."""

import argparse
import csv
import json
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.simulation.privileged_grasp import GraspRig
from humanoid.simulation.tasks.toy_cleanup import ToyCleanupTask, deposited_toys
from humanoid.simulation.video import EpisodeVideo

CUBE = ("cube_0",)
# The original regression commands a 2 mm squeeze beyond the 36 mm cube width.
NOMINAL_CLOSURE = (0.050 - 0.036 + 0.002) / 0.023
APPROACH_OFFSET = np.array([0.0, 0.30])
CARRY_LIFT = 0.18
BASE_SPEED = 0.10
BASE_TOLERANCE = 0.003
BASE_WALL_CLEARANCE = 0.21
BIN_PASSAGE_OFFSET = 0.40
BIN_PARKING_DISTANCE = 0.34
MAX_CLOSURE_RAD = 2.2028


class CubeDelivery:
    def __init__(self, rig: GraspRig, folder: Path, resources: ExitStack, *, video: bool):
        self.rig = rig
        self.engine = rig.engine
        self.data = self.engine.data
        self.cube = self.data.body(CUBE[0])
        self.pinch = self.data.site("pinch")
        self.root = self.engine.binding.root
        assert self.root is not None
        self.phase = "reset"
        self.peak_finger_load = 0.0
        self.peak_overlap = 0.0
        self.carry_contact_steps = 0
        self.carry_steps = 0
        self.peak_shift = 0.0
        self.grasp_offset: np.ndarray | None = None
        self.video = (
            resources.enter_context(
                closing(
                    EpisodeVideo(
                        self.engine,
                        folder / "delivery.mp4",
                        overview=ToyCleanupTask(CUBE).overview_camera(self.data),
                        follow_body=CUBE[0],
                    )
                )
            )
            if video
            else None
        )
        stream = resources.enter_context((folder / "trace.csv").open("w", newline=""))
        self.trace = csv.writer(stream)
        self.trace.writerow(
            (
                "time_s",
                "phase",
                "cube_x_m",
                "cube_y_m",
                "cube_z_m",
                "pinch_x_m",
                "pinch_y_m",
                "pinch_z_m",
                "closure_target_rad",
                "left_N",
                "right_N",
                "support_N",
                "penetration_m",
            )
        )

    def sample(self) -> None:
        loads, penetration = self.rig.contact_metrics(self.cube.id)
        self.peak_finger_load = max(self.peak_finger_load, float(max(loads[:2])))
        self.peak_overlap = max(self.peak_overlap, penetration)
        if self.phase.startswith("carry"):
            offset = self.pinch.xmat.reshape(3, 3).T @ (self.cube.xpos - self.pinch.xpos)
            if self.grasp_offset is None:
                self.grasp_offset = offset
            self.peak_shift = max(
                self.peak_shift, float(np.linalg.norm(offset - self.grasp_offset))
            )
            self.carry_steps += 1
            self.carry_contact_steps += int(min(loads[:2]) > 0 and loads[2] == 0)
        self.trace.writerow(
            (
                self.data.time,
                self.phase,
                *self.cube.xpos,
                *self.pinch.xpos,
                self.data.ctrl[self.rig.gripper],
                *loads,
                penetration,
            )
        )
        if self.video is not None:
            self.video.capture(self.phase)

    def arm(self, phase: str, target: np.ndarray, duration: float) -> None:
        self.phase = phase
        self.rig.move(target, duration, self.sample)

    def drive(self, phase: str, xy: np.ndarray) -> None:
        """Translate at bounded speed while the position actuators hold the arm and grip."""
        self.phase = phase
        # Keep the chassis clear of side walls; the arm can reach a cube near the wall.
        xy = xy.copy()
        xy[0] = np.clip(
            xy[0],
            self.data.geom("wall_left").xpos[0] + BASE_WALL_CLEARANCE,
            self.data.geom("wall_right").xpos[0] - BASE_WALL_CLEARANCE,
        )
        root = self.root
        assert root is not None
        addresses = list(root.qpos_addresses)
        actuators = list(root.actuator_ids)
        for _ in range(round(35 / self.engine.physics_timestep)):
            x, y, yaw = self.data.qpos[addresses]
            delta = xy - (x, y)
            distance = float(np.linalg.norm(delta))
            if distance < BASE_TOLERANCE:
                break
            velocity = delta * min(2.0, BASE_SPEED / distance)
            self.data.ctrl[actuators] = [*velocity, np.clip(-2 * yaw, -0.4, 0.4)]
            self.engine.step()
            self.sample()
        else:
            raise RuntimeError(f"Base stalled during {phase}, {distance:.3f} m from its target.")
        self.engine.stop_velocity_actuators()

    def run(self, closure: float) -> None:
        root = self.root
        assert root is not None
        base_xy = list(root.qpos_addresses[:2])
        pickup = self.cube.xpos[:2].copy() - APPROACH_OFFSET
        bin_xy = self.data.geom("bin_floor").xpos[:2].copy()
        lane = pickup[0]
        detour = pickup[1] > bin_xy[1] and abs(lane - bin_xy[0]) < BIN_PASSAGE_OFFSET
        if detour:
            # Approach cubes behind the bin via a lane wide enough for the chassis.
            room_left = self.data.geom("wall_left").xpos[0] + BASE_WALL_CLEARANCE
            room_right = self.data.geom("wall_right").xpos[0] - BASE_WALL_CLEARANCE
            lanes = bin_xy[0] + np.array([-BIN_PASSAGE_OFFSET, BIN_PASSAGE_OFFSET])
            lane = min(
                (x for x in lanes if room_left < x < room_right), key=lambda x: abs(x - pickup[0])
            )
            pickup[1] = max(pickup[1], bin_xy[1] + BIN_PARKING_DISTANCE)
        # Fold the wrist close to the chassis so the D405 clears the front wall.
        home_reach = self.pinch.xpos[1] - self.data.qpos[base_xy[1]]
        if pickup[1] + home_reach + 0.10 > self.data.geom("wall_front").xpos[1]:
            stow = np.array([*self.data.qpos[base_xy], 0.09]) + np.array([0, 0.16, 0])
            self.arm("stow for travel", stow, 3.0)
        self.drive("approach along x", np.array([lane, self.data.qpos[base_xy[1]]]))
        self.drive("approach along y", np.array([lane, pickup[1]]))
        self.drive("approach cube", pickup)
        target = self.cube.xpos.copy()
        above = target + np.array([0, 0, 0.08])
        self.arm("pregrasp", above, 2.5)
        self.arm("descend", target, 1.5)
        # Use the same closing speed and time window in every condition. Only the
        # final jaw position changes; all commands follow the same ramp until that cap.
        increments = int(np.ceil(MAX_CLOSURE_RAD / (NOMINAL_CLOSURE / 49)))
        for command in np.linspace(0, MAX_CLOSURE_RAD, increments + 1):
            self.data.ctrl[self.rig.gripper] = min(command, closure)
            self.arm("close", target, 0.015)
        self.arm("squeeze", target, 0.7)
        for dz in np.linspace(0, CARRY_LIFT, 180):
            self.arm("lift", target + np.array([0, 0, dz]), 0.02)
        self.arm("hold", target + np.array([0, 0, CARRY_LIFT]), 1)

        # Align with the bin's near edge first, then translate the raised cube over its center.
        if detour:
            # Bring the loaded wrist inside the wall clearance before translating sideways.
            carry = np.array([*self.data.qpos[base_xy], 0.11]) + np.array([0, 0.18, 0])
            self.arm("tuck for travel", carry, 3.0)
            self.drive("carry around bin", np.array([lane, self.data.qpos[base_xy[1]]]))
            self.drive("carry along bin", np.array([lane, bin_xy[1] - APPROACH_OFFSET[1]]))
            reach = np.array([*self.data.qpos[base_xy], CARRY_LIFT]) + np.array([0, 0.30, 0])
            self.arm("extend above rim", reach, 3.0)
        goal = bin_xy - (self.pinch.xpos[:2] - self.data.qpos[base_xy])
        self.drive("carry to bin", np.array([self.data.qpos[base_xy[0]], goal[1]]))
        self.drive("carry over bin", goal)
        # Lower just inside the rim before opening. Keep the arm above the bin floor.
        release = np.array([*bin_xy, self.data.geom("bin_left").xpos[2] + 0.06])
        self.arm("lower into bin", release, 1.5)
        for command in np.linspace(closure, 0, 50):
            self.data.ctrl[self.rig.gripper] = command
            self.arm("release", release, 0.015)
        self.arm("retract", release + np.array([0, 0, 0.08]), 1.5)
        self.drive("retreat", self.data.qpos[base_xy] - np.array([0, 0.10]))
        self.phase = "settle"
        for _ in range(round(1 / self.engine.physics_timestep)):
            self.engine.step()
            self.sample()


def trial(folder: Path, seed: int, closure: float, *, video: bool) -> bool:
    folder.mkdir(parents=True)
    with ExitStack() as resources:
        task = ToyCleanupTask(CUBE)
        rig = GraspRig(TRISKEL_CONFIG, lambda spec: task.configure_scene(spec, seed))
        resources.callback(rig.engine.close)
        rig.engine.step(100)
        delivery = CubeDelivery(rig, folder, resources, video=video)
        cube_start = delivery.cube.xpos.copy()
        error = None
        try:
            delivery.run(closure)
        except RuntimeError as exc:
            error = str(exc)
        loads, _ = rig.contact_metrics(delivery.cube.id)
        deposited = deposited_toys(delivery.data, CUBE)
        cube_geom = int(rig.engine.model.body(CUBE[0]).geomadr[0])
        floor_geom = rig.engine.model.geom("bin_floor").id
        on_floor = any(
            {contact.geom1, contact.geom2} == {cube_geom, floor_geom}
            for contact in delivery.data.contact
        )
        success = bool(
            deposited and on_floor and loads[0] == loads[1] == 0 and loads[2] > 0 and error is None
        )
        result = {
            "seed": seed,
            "closure_rad": closure,
            "cube_start_xyz_m": cube_start.tolist(),
            "bin_floor_xyz_m": delivery.data.geom("bin_floor").xpos.tolist(),
            "cube_final_xyz_m": delivery.cube.xpos.tolist(),
            "on_bin_floor": on_floor,
            "deposited": deposited,
            "success": success,
            "peak_finger_load_N": delivery.peak_finger_load,
            "peak_overlap_mm": 1000 * delivery.peak_overlap,
            "carry_bilateral_fraction": delivery.carry_contact_steps / delivery.carry_steps
            if delivery.carry_steps
            else None,
            "carry_max_slip_mm": 1000 * delivery.peak_shift if delivery.carry_steps else None,
            "final_contact_loads_N": loads.tolist(),
            "error": error,
        }
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--closure", nargs="+", type=float, default=[NOMINAL_CLOSURE])
    parser.add_argument("--output", type=Path, default=Path("logs/privileged-cleanup"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if min(args.seeds) < 0 or any(not 0 <= v <= MAX_CLOSURE_RAD for v in args.closure):
        parser.error("Use nonnegative seeds and Triskel gripper positions in [0, 2.2028] rad.")
    folder = args.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    print(folder.resolve(), flush=True)
    for seed in args.seeds:
        for closure in args.closure:
            trial(folder / f"seed-{seed}-closure-{closure:.6f}", seed, closure, video=args.video)


if __name__ == "__main__":
    main()
