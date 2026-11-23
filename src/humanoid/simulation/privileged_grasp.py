"""Ground-truth pinch-point servo and contact measurements for grasp experiments."""

from collections.abc import Callable
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.simulation import SimulationScene


class GraspRig:
    def __init__(
        self, config: RobotConfig, configure_scene: Callable[[mujoco.MjSpec], None] | None = None
    ) -> None:
        self.triskel = config.name == RobotName.TRISKEL
        frame = "gripper_base_link" if self.triskel else "panda_hand"
        offset = [0.002, 0.127, 0] if self.triskel else [0, 0, 0.104]
        simulation = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=SimulationScene.TOY_CLEANUP)

        def add_marker(spec: mujoco.MjSpec) -> None:
            if configure_scene is not None:
                configure_scene(spec)
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
