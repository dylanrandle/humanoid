"""Triskel's single red, 25 mm cube demonstration defaults."""

from dataclasses import replace

import numpy as np

from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.cube_pick import CubeDetectionMethod, CubePickConfig

CUBE_PICK_CONFIG = CubePickConfig(
    cube_size=0.025,
    ground_height=0.0,
    # gripper_base_link +Y points along the fins. Target near their tips.
    grip_offset=(0.0, 0.125, 0.0),
    approach_clearance=0.03,
    position_tolerance=0.005,
    orientation_tolerance=0.08,
    open_position=0.0,
    closed_position=2.2,
    squeeze_seconds=1.0,
    phase_timeout=15.0,
    image_max_age=0.2,
    minimum_points=20,
    top_tolerance=0.0015,
    workspace_min=(-0.12, 0.20, 0.0),
    workspace_max=(0.12, 0.40, 0.06),
)

# (base-frame x, y, yaw); each trial contains exactly one ground-level cube.
CUBE_PICK_TEST_POSES = (
    (0.0, 0.28, 0.0),
    (-0.04, 0.27, np.deg2rad(20)),
    (0.04, 0.28, np.deg2rad(-25)),
    (-0.025, 0.32, np.deg2rad(40)),
    (0.025, 0.34, np.deg2rad(-40)),
)

CUBE_PICK_SUBSTEPS = 4
CUBE_PICK_CONTROLLER_CONFIG = OperationalSpaceConfig(
    dt=CUBE_PICK_SUBSTEPS * DEFAULT_MUJOCO_SIMULATION_CONFIG.physics_timestep,
    tool_position_cost=5,
    tool_orientation_cost=1,
    manipulability_cost=0,
    damping_cost=0.05,
    joint_velocity_limit=1,
    avoid_collisions=False,
)

# The base origin is at the lower plate. Wheel meshes extend 32 mm below it.
# A 45 mm block needs only a small closure from the roughly 50 mm open gap:
# the URDF mimic joints move each fin 11.5 mm per actuator radian.
WOOD_CUBE_PICK_CONFIG = replace(
    CUBE_PICK_CONFIG,
    cube_size=0.045,
    ground_height=-0.032,
    grip_offset=(0.002, 0.125, 0.0),
    closed_position=0.30,
    minimum_points=100,
    top_tolerance=0.004,
    workspace_min=(-0.08, 0.30, -0.032),
    workspace_max=(0.08, 0.35, 0.03),
    detection_method=CubeDetectionMethod.DEPTH_SIDES,
)
WOOD_CUBE_TEST_POSES = (
    (0.0, 0.31, 0.0),
    (-0.025, 0.31, np.deg2rad(20)),
    (0.02, 0.32, np.deg2rad(-25)),
    (-0.025, 0.34, np.deg2rad(40)),
    (0.025, 0.335, np.deg2rad(-40)),
)
