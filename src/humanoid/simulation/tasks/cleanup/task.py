"""Privileged mobile pick-and-place expert; physical grasps and verified deposits."""

from dataclasses import asdict, replace
from functools import partial

import numpy as np
import pinocchio as pin

from humanoid.config.tasks.cleanup import (
    CLEANUP_SIMULATION_CONFIG,
    TRISKEL_CLEANUP_CONFIG,
    TRISKEL_CLEANUP_SCENE,
    TRISKEL_CLEANUP_VIEW,
    TRISKEL_MOBILE_CLEANUP_CONFIG,
    WRIST_CAMERA_NAME,
)
from humanoid.logger import get_logger
from humanoid.simulation.client import SimulationClient
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.objects import body_bounding_box, read_object
from humanoid.simulation.tasks.base import SimulationTask
from humanoid.simulation.tasks.cleanup.motion import CleanupMotion, grasp_rotation
from humanoid.simulation.tasks.cleanup.navigation import inflated_obstacle, shortest_approach
from humanoid.simulation.tasks.cleanup.scene import add_cleanup_scene
from humanoid.types.camera import RGBCameraConfig
from humanoid.types.collection import TaskResult
from humanoid.types.object_state import ObjectState
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.tasks.cleanup import NavigationMap

logger = get_logger(__name__)
MAX_LAYOUT_JITTER = 0.008
SETTLED_LINEAR_SPEED = 0.025
SETTLED_ANGULAR_SPEED = 0.5
CONTAINMENT_TOLERANCE = 0.002
MAX_HELD_OFFSET = 0.06


class CleanupTask(SimulationTask):
    name = "cleanup"
    version = 9
    label = "Cleanup"
    robot = RobotName.TRISKEL

    def __init__(self, jitter: float = 0.006, *, robot_config: RobotConfig | None = None):
        super().__init__(robot_config)
        if not np.isfinite(jitter) or not 0 <= jitter <= MAX_LAYOUT_JITTER:
            raise ValueError(f"Layout jitter must be in [0, {MAX_LAYOUT_JITTER}] meters.")
        self.jitter = jitter
        self.scene = TRISKEL_CLEANUP_SCENE
        self.arm_config = TRISKEL_CLEANUP_CONFIG
        self.mobile_config = TRISKEL_MOBILE_CLEANUP_CONFIG
        self.view_config = TRISKEL_CLEANUP_VIEW
        self.bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.attempts = 0
        self.motion: CleanupMotion | None = None

    def create_engine(self, seed: int) -> NativeMujocoEngine:
        random = np.random.default_rng(seed)
        self.scene = replace(
            TRISKEL_CLEANUP_SCENE,
            objects=tuple(
                replace(
                    item,
                    position=tuple(
                        np.asarray(item.position) + random.uniform(-self.jitter, self.jitter, 2)
                    ),
                    yaw=item.yaw + float(random.uniform(-0.3, 0.3)),
                )
                for item in TRISKEL_CLEANUP_SCENE.objects
            ),
        )
        engine = NativeMujocoEngine(
            self.robot_config,
            CLEANUP_SIMULATION_CONFIG,
            scene_builder=partial(add_cleanup_scene, config=self.scene),
        )
        self.bounds = {
            item.name: body_bounding_box(engine, item.name) for item in self.scene.objects
        }
        self.attempts = 0
        self.motion = None
        return engine

    def cameras(self) -> dict[str, RGBCameraConfig]:
        return self.scene.cameras

    def objects(self, engine: NativeMujocoEngine) -> tuple[ObjectState, ...]:
        return tuple(read_object(engine, name, bounds) for name, bounds in self.bounds.items())

    def metadata(self) -> dict[str, object]:
        return {
            "scene": asdict(self.scene),
            "arm": asdict(self.arm_config),
            "mobile": asdict(self.mobile_config),
            "simulation": asdict(CLEANUP_SIMULATION_CONFIG),
            "robot": {
                "name": self.robot_config.name.value,
                "gripper": asdict(self.robot_config.gripper) if self.robot_config.gripper else None,
            },
            "view": asdict(self.view_config),
            "policy_observation": "current_four_camera_rgb_only",
            "visibility_policy": "diagnostic_only",
            "layout_jitter_m": self.jitter,
        }

    def state(self, runtime: SimulationClient) -> dict[str, object]:
        return {
            "phase": runtime.phase,
            "target": runtime.target,
            "attempts": self.attempts,
            "contained": [item.name for item in self.objects(runtime.engine) if self.in_bin(item)],
            "bin_position": self.scene.bin_position,
            "bin_interior_half_size": (
                np.asarray(self.scene.bin_half_size) - self.scene.wall_thickness
            ).tolist(),
            "base_distance_m": self.motion.base_distance if self.motion else 0.0,
            "planned_distance_m": self.motion.planned_distance if self.motion else 0.0,
            "navigation": {
                "target": self.motion.navigation_plan.target.tolist(),
                "waypoints": [point.tolist() for point in self.motion.navigation_plan.waypoints],
                "approach_yaw": self.motion.navigation_plan.yaw,
                "length_m": self.motion.navigation_plan.length,
            }
            if self.motion and self.motion.navigation_plan
            else None,
            "view_target": self.motion.view_target if self.motion else "",
            "visibility": asdict(self.motion.view.best_view(self.motion.view_target))
            if self.motion and self.motion.view_target
            else None,
            "views": {name: asdict(view) for name, view in self.motion.views().items()}
            if self.motion
            else {},
        }

    def in_bin(self, item: ObjectState) -> bool:
        corners = item.corners
        interior = np.asarray(self.scene.bin_half_size) - self.scene.wall_thickness
        bottom = self.scene.floor_height + self.scene.wall_thickness
        rim = self.scene.floor_height + self.scene.bin_height
        return bool(
            np.all(
                np.abs(corners[:, :2] - self.scene.bin_position) <= interior + CONTAINMENT_TOLERANCE
            )
            and corners[:, 2].min() >= bottom - CONTAINMENT_TOLERANCE
            and corners[:, 2].max() <= rim + CONTAINMENT_TOLERANCE
            and item.linear_velocity is not None
            and item.angular_velocity is not None
            and np.linalg.norm(item.linear_velocity) < SETTLED_LINEAR_SPEED
            and np.linalg.norm(item.angular_velocity) < SETTLED_ANGULAR_SPEED
        )

    def success(self, engine: NativeMujocoEngine) -> bool:
        return bool(self.bounds) and all(self.in_bin(item) for item in self.objects(engine))

    def navigation_map(self, engine: NativeMujocoEngine, carried: str = "") -> NavigationMap:
        config = self.mobile_config
        radius = config.footprint_radius + config.clearance
        bin_corners = (
            np.asarray(self.scene.bin_position)
            + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.scene.bin_half_size
        )
        obstacles = [inflated_obstacle(bin_corners, radius, config.obstacle_sides)]
        for item in self.objects(engine):
            if item.name == carried:
                continue
            obstacles.append(inflated_obstacle(item.corners[:, :2], radius, config.obstacle_sides))
        return NavigationMap(
            np.asarray(config.navigation_min) + radius,
            np.asarray(config.navigation_max) - radius,
            tuple(obstacles),
        )

    def run(self, runtime: SimulationClient) -> TaskResult:
        motion = self.motion = CleanupMotion(
            runtime, self.arm_config, self.mobile_config, self.scene, self.view_config
        )
        try:
            runtime.phase = "prepare"
            motion.grip(False)
            motion.park()
            motion.hold(0.5)
            while remaining := [
                item.name for item in self.objects(runtime.engine) if not self.in_bin(item)
            ]:
                runtime.phase, runtime.target = "search", ""
                name = motion.acquire(remaining)
                while not self.in_bin(self._item(runtime.engine, name)):
                    if self.attempts >= self.arm_config.max_attempts:
                        return TaskResult(False, "Grasp attempt budget exhausted.")
                    self.attempts += 1
                    runtime.target = name
                    logger.info("Cleanup attempt %s: %s", self.attempts, name)
                    if self._fetch(runtime, motion, name):
                        break
                if not self.in_bin(self._item(runtime.engine, name)):
                    return TaskResult(
                        False, f"Deposit of {name} failed full-box containment/settling checks."
                    )
            runtime.phase, runtime.target = "verify", ""
            motion.hold(1.0)
            # Require success to persist, not merely a crossing of the goal volume.
            for _ in range(5):
                if not self.success(runtime.engine):
                    return TaskResult(False, "An object is outside the bin or has not settled.")
                motion.hold(0.2)
            return TaskResult(True, "All task objects fully contained and settled in the bin.")
        except (ValueError, TimeoutError) as error:
            return TaskResult(False, str(error))
        finally:
            runtime.stop()

    def _item(self, engine: NativeMujocoEngine, name: str) -> ObjectState:
        return read_object(engine, name, self.bounds[name])

    def _navigate(
        self,
        runtime: SimulationClient,
        motion: CleanupMotion,
        target: np.ndarray,
        carried: str = "",
        approach_yaws: np.ndarray | None = None,
    ) -> None:
        plan = shortest_approach(
            self.navigation_map(runtime.engine, carried),
            motion.base_pose()[:2],
            target,
            np.asarray(self.mobile_config.pickup_offset),
            approach_yaws=approach_yaws,
        )
        motion.drive(plan)
        if carried and not self._holding(runtime.engine, motion, carried):
            raise ValueError(f"Lost {carried} during transport.")

    def _holding(self, engine: NativeMujocoEngine, motion: CleanupMotion, name: str) -> bool:
        item = self._item(engine, name)
        grasp = motion.kinematics.grasp_pose(motion.positions()).translation
        return bool(
            motion.aperture() > self.arm_config.min_holding_width
            and item.corners[:, 2].min() > self.scene.floor_height + 0.035
            and np.linalg.norm(item.pose.translation - grasp) < MAX_HELD_OFFSET
        )

    def _drop_center(self, engine: NativeMujocoEngine) -> np.ndarray:
        """Choose the leftmost empty bin slot in the current camera image.

        Slot occupancy is measured from the present scene, not the pickup index;
        changing the visible-object collection order cannot overwrite a deposit.
        """
        slots = np.asarray(self.arm_config.drop_slot_offsets) + self.scene.bin_position
        occupied = {
            int(np.argmin(np.linalg.norm(slots - item.pose.act(item.bbox_center)[:2], axis=1)))
            for item in self.objects(engine)
            if self.in_bin(item)
        }
        candidates = np.array([slot for index, slot in enumerate(slots) if index not in occupied])
        if not len(candidates):
            raise ValueError("No empty drop slot remains in the bin.")
        camera = engine.data.camera(WRIST_CAMERA_NAME)
        points = np.c_[candidates, np.full(len(candidates), self.scene.floor_height)]
        local = (points - camera.xpos) @ (camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1]))
        index = int(np.argmin(local[:, 0] / local[:, 2]))
        return np.r_[candidates[index], self.scene.floor_height + self.arm_config.release_height]

    def _fetch(self, runtime: SimulationClient, motion: CleanupMotion, name: str) -> bool:
        runtime.phase = "approach"
        item = self._item(runtime.engine, name)
        object_yaw = np.arctan2(item.pose.rotation[1, 0], item.pose.rotation[0, 0])
        headings = object_yaw + np.arange(4) * np.pi / 2
        self._navigate(runtime, motion, item.pose.translation[:2], approach_yaws=headings)
        motion.view_target = name
        item = self._item(runtime.engine, name)
        base_yaw = motion.base_pose()[2]
        # Align the fingers with a box face for a stable parallel-jaw grasp.
        rotation = grasp_rotation(base_yaw)
        width = np.ptp(item.corners @ rotation[:, 0])
        if width >= self.arm_config.open_width - 0.002:
            raise ValueError("Object is too wide for a grasp from this approach.")
        center = item.pose.act(item.bbox_center)
        above = pin.SE3(
            rotation,
            np.array([*center[:2], self.scene.floor_height + self.arm_config.transit_height]),
        )
        grasp = pin.SE3(
            rotation, np.array([*center[:2], max(center[2], self.scene.floor_height + 0.017)])
        )
        runtime.phase = "grasp"
        motion.grip(False)
        motion.reach(above)
        motion.move(grasp)
        motion.grip(True)
        runtime.phase = "lift"
        motion.move(above)
        if not self._holding(runtime.engine, motion, name):
            motion.grip(False)
            motion.park()
            return False
        runtime.phase = "search_bin"
        motion.carried_target = name
        motion.acquire(["cleanup_bin"])
        runtime.phase = "transport"
        self._navigate(
            runtime,
            motion,
            np.asarray(self.scene.bin_position),
            carried=name,
        )
        runtime.phase = "place"
        # Use the measured object-in-gripper transform. Grasp offsets and small
        # contact slips rotate with the new approach heading; treating the tool
        # center as the object center can otherwise place a toy on its neighbor.
        center = self._drop_center(runtime.engine)
        item = self._item(runtime.engine, name)
        held = motion.kinematics.grasp_pose(motion.positions()).inverse() * pin.SE3(
            item.pose.rotation, item.pose.act(item.bbox_center)
        )
        expected_rotation = grasp_rotation(motion.base_pose()[2]) @ held.rotation
        object_yaw = np.arctan2(expected_rotation[1, 0], expected_rotation[0, 0])
        desired_object = pin.SE3(pin.exp3(np.array([0.0, 0.0, object_yaw])), center)
        # Re-observe after the longer transfer, then correct residual slip before
        # releasing. The object is moved exclusively by actuator/contact physics.
        for correction in range(2):
            item = self._item(runtime.engine, name)
            held = motion.kinematics.grasp_pose(motion.positions()).inverse() * pin.SE3(
                item.pose.rotation, item.pose.act(item.bbox_center)
            )
            if correction == 0:
                motion.reach(desired_object * held.inverse())
            else:
                motion.move(desired_object * held.inverse())
        if not self._holding(runtime.engine, motion, name):
            raise ValueError(f"Lost {name} while approaching the bin.")
        motion.grip(False)
        motion.carried_target = ""
        runtime.phase = "inspect_bin"
        motion.park()
        motion.hold(0.8)
        return True
