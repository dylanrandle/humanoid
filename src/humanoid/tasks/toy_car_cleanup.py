"""A room, one open bin, and freely moving cars; no grasp welds or object teleport tools."""

import itertools

import mujoco
import numpy as np

from humanoid.config.tasks import TOY_CAR_CLEANUP_CONFIG
from humanoid.types.tasks import TaskEvaluation, ToyCarCleanupConfig

EXTERNAL_CONTACT = 2
ROBOT_AND_EXTERNAL_CONTACT = 3
MAX_SETTLED_SPEED = 0.02
ROBOT_SPAWN_CLEARANCE = 0.40
CAR_SPAWN_CLEARANCE = 0.15


class ToyCarCleanupTask:
    name = "toy_car_cleanup"
    instruction = (
        "Find every toy car in the room. Pick up and carry each car individually, "
        "return to the bin, and release it into the bin. Continue searching until "
        "all cars have been deposited. Do not push cars across the floor as a substitute "
        "for carrying them. Report uncertainty or failure rather than false completion."
    )

    def __init__(self, config: ToyCarCleanupConfig = TOY_CAR_CLEANUP_CONFIG):
        self.config = config
        self._inside_since: list[float | None] = [None] * config.car_count
        self._lifted = [False] * config.car_count
        self._multiple_carry = False

    def build_scene(self, spec: mujoco.MjSpec) -> None:
        c = self.config
        world = spec.worldbody
        world.add_geom(
            name="task_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, c.ground_z],
            size=[2, 2, 0.01],
            rgba=[0.45, 0.45, 0.45, 1],
            contype=EXTERNAL_CONTACT,
            conaffinity=ROBOT_AND_EXTERNAL_CONTACT,
        )
        world.add_light(pos=[0, 0.7, 2], dir=[0, 0, -1], ambient=[0.3, 0.3, 0.3])
        room_center = (np.asarray(c.room_min) + c.room_max) / 2
        room_half = (np.asarray(c.room_max) - c.room_min) / 2
        for axis in range(2):
            for sign in (-1, 1):
                center = [*room_center, c.ground_z + 0.20]
                size = [*room_half, 0.20]
                center[axis] += sign * room_half[axis]
                size[axis] = 0.015
                self._box(world, f"wall_{axis}_{sign}", center, size, [0.6, 0.65, 0.7, 1])
        self._build_bin(world)
        for index in range(c.car_count):
            self._build_car(world, index)

    @staticmethod
    def _box(body, name, position, half_size, rgba, **kwargs):
        return body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=position,
            size=half_size,
            rgba=rgba,
            contype=EXTERNAL_CONTACT,
            conaffinity=ROBOT_AND_EXTERNAL_CONTACT,
            friction=[0.6, 0.005, 0.0001],
            **kwargs,
        )

    def _build_bin(self, world) -> None:
        c = self.config
        x, y = c.bin_center
        hx, hy = np.asarray(c.bin_inner_size) / 2
        t = c.bin_wall_thickness
        self._box(
            world,
            "bin_floor",
            [x, y, c.ground_z + t / 2],
            [hx + t, hy + t, t / 2],
            [0.15, 0.35, 0.65, 1],
        )
        for axis, half in enumerate((hx, hy)):
            for sign in (-1, 1):
                center = [x, y, c.ground_z + c.bin_rim_height / 2]
                size = [hx + t, hy + t, c.bin_rim_height / 2]
                center[axis] += sign * (half + t / 2)
                size[axis] = t / 2
                self._box(world, f"bin_wall_{axis}_{sign}", center, size, [0.15, 0.35, 0.65, 1])

    def _build_car(self, world, index: int) -> None:
        c = self.config
        car = world.add_body(name=f"car_{index}", pos=[0, 0.5, c.ground_z + c.car_height / 2])
        car.add_freejoint(name=f"car_{index}_free")
        color = [0.8, 0.2 + 0.15 * (index % 3), 0.1, 1]
        self._box(
            car,
            f"car_{index}_chassis",
            [0, 0, 0],
            [c.car_length / 2, c.car_width * 0.40, c.car_height * 0.20],
            color,
            mass=c.car_mass * 0.65,
        )
        self._box(
            car,
            f"car_{index}_cabin",
            [0, 0, c.car_height * 0.30],
            [c.car_length * 0.24, c.car_width * 0.36, c.car_height * 0.20],
            color,
            mass=c.car_mass * 0.25,
        )
        for wheel, (sx, sy) in enumerate(itertools.product((-1, 1), repeat=2)):
            body = car.add_body(
                name=f"car_{index}_wheel_{wheel}",
                pos=[sx * c.car_length * 0.30, sy * c.car_width * 0.44, -c.car_height * 0.20],
            )
            body.add_joint(
                name=f"car_{index}_wheel_joint_{wheel}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 1, 0],
                damping=0.00001,
            )
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[c.car_height * 0.30, c.car_width * 0.06, 0],
                quat=[np.sqrt(0.5), np.sqrt(0.5), 0, 0],
                rgba=[0.08, 0.08, 0.08, 1],
                mass=c.car_mass * 0.025,
                contype=EXTERNAL_CONTACT,
                conaffinity=ROBOT_AND_EXTERNAL_CONTACT,
                friction=[0.8, 0.001, 0.0001],
            )

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> None:
        c = self.config
        rng = np.random.default_rng(seed)
        placed = []
        for index in range(c.car_count):
            for _ in range(1000):
                xy = rng.uniform(np.asarray(c.room_min) + 0.12, np.asarray(c.room_max) - 0.12)
                away_from_robot = np.linalg.norm(xy) > ROBOT_SPAWN_CLEARANCE
                away_from_bin = np.any(
                    np.abs(xy - c.bin_center) > np.asarray(c.bin_inner_size) / 2 + 0.12
                )
                if (
                    away_from_robot
                    and away_from_bin
                    and all(np.linalg.norm(xy - p) > CAR_SPAWN_CLEARANCE for p in placed)
                ):
                    break
            else:
                raise ValueError("Cannot place cars without overlap in the configured room.")
            placed.append(xy)
            yaw = rng.uniform(-np.pi, np.pi)
            address = int(model.joint(f"car_{index}_free").qposadr[0])
            data.qpos[address : address + 7] = [
                *xy,
                c.ground_z + c.car_height / 2 + 0.002,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
        self._inside_since = [None] * c.car_count
        self._lifted = [False] * c.car_count
        self._multiple_carry = False
        mujoco.mj_forward(model, data)

    def evaluate(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvaluation:
        """Called each physics step by the host; counts fully contained, released cars.

        Geometry is a conservative bounding box including wheels. A historical
        lift and settling are required; position-only or gripper-close events do
        not pass. This is a simulation oracle, not a real-world perception result.
        """
        c = self.config
        delivered = 0
        carried = 0
        corners = (
            np.array(list(itertools.product((-1, 1), repeat=3)))
            * np.array([c.car_length, c.car_width, c.car_height])
            / 2
        )
        for index in range(c.car_count):
            body_id = model.body(f"car_{index}").id
            position = data.xpos[body_id]
            pts = corners @ data.xmat[body_id].reshape(3, 3).T + position
            lifted = bool(pts[:, 2].min() > c.ground_z + 0.02)
            lower = np.array(
                [
                    *(np.asarray(c.bin_center) - np.asarray(c.bin_inner_size) / 2),
                    c.ground_z + c.bin_wall_thickness - 0.002,
                ]
            )
            upper = np.array(
                [
                    *(np.asarray(c.bin_center) + np.asarray(c.bin_inner_size) / 2),
                    c.ground_z + c.bin_rim_height,
                ]
            )
            inside = bool(np.all(pts >= lower) and np.all(pts <= upper))
            dof = int(model.joint(f"car_{index}_free").dofadr[0])
            settled = np.linalg.norm(data.qvel[dof : dof + 6]) < MAX_SETTLED_SPEED
            # Robot-car contact prevents treating a held object as deposited.
            robot_contact = any(
                self._is_robot_contact(model, data.contact[k], body_id) for k in range(data.ncon)
            )
            if lifted and robot_contact:
                carried += 1
                self._lifted[index] = True
            if inside and settled and self._lifted[index] and not robot_contact:
                if self._inside_since[index] is None:
                    self._inside_since[index] = float(data.time)
                if data.time - self._inside_since[index] >= c.settle_seconds:
                    delivered += 1
            else:
                self._inside_since[index] = None
        self._multiple_carry |= carried > 1
        complete = delivered == c.car_count and not self._multiple_carry
        return TaskEvaluation(
            task_complete=complete,
            objects_delivered=delivered,
            objects_total=c.car_count,
            reason="complete"
            if complete
            else "multiple_cars_lifted"
            if self._multiple_carry
            else "cars_remaining",
        )

    @staticmethod
    def _is_robot_contact(model: mujoco.MjModel, contact, car_body: int) -> bool:
        roots = [int(model.body_rootid[model.geom_bodyid[g]]) for g in contact.geom]
        return car_body in roots and any(
            root != car_body and model.body(root).name == "base_link" for root in roots
        )
