"""Seeded layouts and simulator-side scoring for the toy cleanup scene."""

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid.simulation.model import ROBOT_CONTACT_TYPE
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene
from humanoid.types.trial import SimulationTask, TaskScore

MIN_TOY_SPACING_M = 0.09
TOYS = tuple(f"{kind}_{i}" for kind in ("car", "cube", "cylinder", "ball") for i in range(3))


def randomize_cleanup(spec: mujoco.MjSpec, seed: int, toys: tuple[str, ...] = TOYS) -> None:
    """Configure the scene before compilation so rendering and contacts share geometry."""
    rng = np.random.default_rng(seed)
    width, length = rng.uniform((1.4, 1.7), (2.0, 2.4))
    left, right = -width / 2, width / 2
    back, front = -0.4, length - 0.4
    wall_color = [*rng.uniform(0.25, 0.9, 3), 1]
    for side, x, y in (
        ("left", left, (back + front) / 2),
        ("right", right, (back + front) / 2),
        ("back", 0, back),
        ("front", 0, front),
    ):
        wall = spec.geom(f"wall_{side}")
        wall.pos[:2] = (x, y)
        if side in ("left", "right"):
            wall.size[1] = length / 2
        else:
            wall.size[0] = width / 2
        wall.rgba = wall_color
    spec.geom("task_floor").rgba = [*rng.uniform(0.18, 0.75, 3), 1]
    bin_pos = rng.uniform((left + 0.18, 0.43), (right - 0.18, front - 0.16))
    spec.geom("bin_floor").frame.pos[:2] = bin_pos
    bin_color = [*rng.uniform(0.1, 0.95, 3), 1]
    for name in ("floor", "left", "right", "back", "front"):
        spec.geom(f"bin_{name}").rgba = bin_color

    occupied = []
    for name in TOYS:
        if name not in toys:
            spec.delete(spec.body(name))
            continue
        # The forward sampling region clears the robot at its home pose.
        for _ in range(1000):
            xy = rng.uniform((left + 0.08, 0.40), (right - 0.08, front - 0.08))
            if np.all(abs(xy - bin_pos) < (0.19, 0.16)):
                continue
            if all(np.linalg.norm(xy - other) > MIN_TOY_SPACING_M for other in occupied):
                break
        else:
            raise RuntimeError("Could not place cleanup toys without overlaps.")
        occupied.append(xy)
        body = spec.body(name)
        # Each replicated body has a local offset frame in the source scene.
        body.pos[:2] = xy - body.frame.pos[:2]
        rgba = [*rng.uniform(0.08, 0.95, 3), 1]
        for geom in body.geoms:
            geom.rgba = rgba


def _inside_bin(data: mujoco.MjData, toys: tuple[str, ...]) -> dict[int, str]:
    origin = data.geom("bin_floor").xpos
    half_width = origin[0] - data.geom("bin_left").xpos[0] - data.model.geom("bin_left").size[0]
    half_depth = origin[1] - data.geom("bin_back").xpos[1] - data.model.geom("bin_back").size[1]
    floor = origin[2] + data.model.geom("bin_floor").size[2]
    rim = data.geom("bin_left").xpos[2] + data.model.geom("bin_left").size[2]
    result = {}
    for name in toys:
        body = data.body(name)
        x, y, z = body.xpos
        if abs(x - origin[0]) < half_width and abs(y - origin[1]) < half_depth and floor < z < rim:
            result[body.id] = name
    return result


def deposited_toys(data: mujoco.MjData, toys: tuple[str, ...] = TOYS) -> list[str]:
    """Count released toys supported by the bin floor, including stacks and wheeled toys."""
    inside = _inside_bin(data, toys)
    model = data.model
    floor = model.geom("bin_floor").id
    held = set()
    on_floor = set()
    supports: dict[int, set[int]] = defaultdict(set)
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact):
        first, second = contact.geom
        a, b = model.body_rootid[model.geom_bodyid[contact.geom]]
        if contact.efc_address < 0 or (a not in inside and b not in inside):
            continue
        if model.geom_contype[first] == ROBOT_CONTACT_TYPE:
            held.add(b)
        if model.geom_contype[second] == ROBOT_CONTACT_TYPE:
            held.add(a)

        # The contact normal points from the first geom to the second.
        vertical = contact.frame[2]
        if vertical == 0:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        if wrench[0] <= 0:
            continue
        lower_geom, lower, upper = (first, a, b) if vertical > 0 else (second, b, a)
        if lower_geom == floor:
            on_floor.add(upper)
        elif lower in inside and upper in inside:
            supports[lower].add(upper)

    supported = on_floor - held
    pending = list(supported)
    while pending:
        above = supports[pending.pop()] - held - supported
        supported.update(above)
        pending.extend(above)
    return [name for body, name in inside.items() if body in supported]


@dataclass(frozen=True)
class ToyCleanupTask(SimulationTask):
    toys: tuple[str, ...] = ("cube_0",)
    robot: RobotName = RobotName.TRISKEL
    scene = SimulationScene.TOY_CLEANUP
    completion_reason = "all_toys_deposited"

    @property
    def goal(self) -> str:
        objective = (
            "Place the one toy cube in the bin."
            if self.toys == ("cube_0",)
            else f"Place all {len(self.toys)} toys in the bin."
        )
        if self.robot is RobotName.TRISKEL:
            objective += (
                " Triskel has a bottom-plate D435 and a wrist-mounted D405. Use both RGB-D views."
            )
        return (
            f"{objective} "
            "Top-down picks worked in our cube tests. Try approaching the cube from above. "
            "Closing too tightly can destabilize the grasp. Close gradually. "
            "Search the whole room and verify deposits after release."
        )

    def configure_scene(self, spec: mujoco.MjSpec, seed: int) -> None:
        randomize_cleanup(spec, seed, self.toys)
        if self.robot is RobotName.PANDA and self.toys == ("cube_0",):
            # The fixed base cannot traverse the room. Sample either side of the
            # hand, with enough lateral clearance between the cube and the bin.
            rng = np.random.default_rng([seed, 1])
            side = rng.choice([-1, 1])
            cube = spec.body("cube_0")
            cube.pos[:2] = [rng.uniform(0.26, 0.34), side * rng.uniform(0.14, 0.23)]
            cube.pos[:2] -= cube.frame.pos[:2]
            spec.geom("bin_floor").frame.pos[:2] = [
                rng.uniform(0.26, 0.34),
                -side * rng.uniform(0.14, 0.23),
            ]
            spec.geom("task_floor").frame.pos[2] = 0

    def score(self, data: mujoco.MjData) -> TaskScore:
        deposited = deposited_toys(data, self.toys)
        return TaskScore(
            len(deposited) == len(self.toys), {"deposited": deposited, "total": len(self.toys)}
        )


def add_arguments(
    parser: argparse.ArgumentParser,
) -> Callable[[argparse.Namespace], ToyCleanupTask]:
    parser.add_argument(
        "--items",
        type=int,
        choices=(1, len(TOYS)),
        default=1,
        help="Use one cube or the full toy set.",
    )
    return _from_arguments


def _from_arguments(args: argparse.Namespace) -> ToyCleanupTask:
    return ToyCleanupTask(("cube_0",) if args.items == 1 else TOYS, RobotName(args.robot))
