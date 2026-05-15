"""Tests for humanoid.robots.base.Robot."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.robots.base import Robot


@pytest.fixture(scope="module")
def panda_robot() -> Robot:
    """Fixed-base 7-DOF Panda — has a separate `panda_collision.urdf`."""
    return Robot(ROBOT_CONFIGS["panda"])


@pytest.fixture(scope="module")
def mobile_robot() -> Robot:
    """Mobile manipulator — planar root joint and continuous wheel joints."""
    return Robot(ROBOT_CONFIGS["elrobot_mobile"])


class TestConstruction:
    def test_loads_fixed_base_robot(self, panda_robot):
        assert panda_robot.model.nq > 0
        assert panda_robot.model.nv > 0
        assert panda_robot.model.njoints > 1

    def test_loads_mobile_robot_with_planar_root(self, mobile_robot):
        """elrobot_mobile has base_frame configured -> planar root joint with nq=4, nv=3."""
        expected_planar_nq = 4
        expected_planar_nv = 3
        root_joint = mobile_robot.model.joints[1]
        assert root_joint.shortname() == "JointModelPlanar"
        assert root_joint.nq == expected_planar_nq
        assert root_joint.nv == expected_planar_nv

    def test_no_planar_root_when_base_frame_missing(self, panda_robot):
        """Without base_frame, no planar root joint is added."""
        root_joint = panda_robot.model.joints[1]
        assert root_joint.shortname() != "JointModelPlanar"

    def test_auto_detects_separate_collision_urdf(self, panda_robot):
        """panda has a `panda_collision.urdf` next to `panda.urdf` and should pick it up."""
        # The collision model should still be populated.
        assert panda_robot.collision_model.ngeoms > 0

    def test_data_structures_created(self, panda_robot):
        assert panda_robot.data is not None
        assert isinstance(panda_robot.collision_data, pin.GeometryData)
        assert isinstance(panda_robot.visual_data, pin.GeometryData)

    def test_missing_robot_dir_raises(self):
        config = replace(ROBOT_CONFIGS["panda"], name="does_not_exist")
        with pytest.raises(FileNotFoundError, match="Robot directory not found"):
            Robot(config)

    def test_missing_urdf_raises(self):
        with pytest.raises(FileNotFoundError, match="URDF file not found"):
            Robot(ROBOT_CONFIGS["panda"], urdf_filename="missing.urdf")

    def test_missing_collision_urdf_raises(self):
        with pytest.raises(FileNotFoundError, match="Collision URDF file not found"):
            Robot(ROBOT_CONFIGS["panda"], collision_urdf_filename="missing_collision.urdf")

    def test_explicit_package_dirs_preserved(self, tmp_path):
        extra = tmp_path / "extra_pkg"
        extra.mkdir()
        robot = Robot(ROBOT_CONFIGS["panda"], package_dirs=[extra])
        # Stored package_dirs include the robot/asset dirs followed by the extras.
        assert robot.package_dirs and extra in robot.package_dirs


class TestProperties:
    def test_config_returned(self, panda_robot):
        assert panda_robot.config is ROBOT_CONFIGS["panda"]

    def test_urdf_path_points_to_existing_file(self, panda_robot):
        assert panda_robot.urdf_path.exists()
        assert panda_robot.urdf_path.name == "panda.urdf"

    def test_srdf_path_constructed(self, panda_robot):
        """srdf_path is computed from the config name even if the file doesn't exist."""
        assert isinstance(panda_robot.srdf_path, Path)
        assert panda_robot.srdf_path.name == "panda.srdf"

    def test_robot_dir_points_to_assets(self, panda_robot):
        assert panda_robot.robot_dir.exists()
        assert panda_robot.robot_dir.name == "panda"

    def test_joint_names_excludes_universe(self, panda_robot):
        names = panda_robot.joint_names
        assert "universe" not in names
        assert len(names) == panda_robot.model.njoints - 1

    def test_joint_names_match_model(self, panda_robot):
        expected = [panda_robot.model.names[i] for i in range(1, panda_robot.model.njoints)]
        assert panda_robot.joint_names == expected


class TestGetFrameId:
    def test_known_frame(self, panda_robot):
        frame_name = panda_robot.config.tool_frame
        frame_id = panda_robot.get_frame_id(frame_name)
        assert panda_robot.model.frames[frame_id].name == frame_name

    def test_unknown_frame_raises(self, panda_robot):
        with pytest.raises(ValueError, match="not found in model"):
            panda_robot.get_frame_id("not_a_real_frame")

    def test_unknown_frame_error_lists_available(self, panda_robot):
        with pytest.raises(ValueError, match="Available frames"):
            panda_robot.get_frame_id("not_a_real_frame")


class TestForwardKinematics:
    def test_populates_frame_placements(self, panda_robot):
        q = pin.neutral(panda_robot.model)
        panda_robot.forward_kinematics(q)

        frame_id = panda_robot.get_frame_id(panda_robot.config.tool_frame)
        pose = panda_robot.data.oMf[frame_id]
        assert isinstance(pose, pin.SE3)

    def test_different_q_changes_frame_pose(self, panda_robot):
        frame_id = panda_robot.get_frame_id(panda_robot.config.tool_frame)

        panda_robot.forward_kinematics(pin.neutral(panda_robot.model))
        pose_neutral = panda_robot.data.oMf[frame_id].copy()

        panda_robot.forward_kinematics(panda_robot.config.home_position)
        pose_home = panda_robot.data.oMf[frame_id].copy()

        assert not np.allclose(pose_neutral.translation, pose_home.translation)


class TestGetFramePose:
    def test_returns_se3(self, panda_robot):
        pose = panda_robot.get_frame_pose(
            panda_robot.config.tool_frame, panda_robot.config.home_position
        )
        assert isinstance(pose, pin.SE3)

    def test_matches_forward_kinematics(self, panda_robot):
        q = panda_robot.config.home_position
        pose_direct = panda_robot.get_frame_pose(panda_robot.config.tool_frame, q)

        panda_robot.forward_kinematics(q)
        frame_id = panda_robot.get_frame_id(panda_robot.config.tool_frame)
        pose_via_fk = panda_robot.data.oMf[frame_id]

        np.testing.assert_allclose(pose_direct.translation, pose_via_fk.translation)
        np.testing.assert_allclose(pose_direct.rotation, pose_via_fk.rotation)

    def test_unknown_frame_raises(self, panda_robot):
        with pytest.raises(ValueError, match="not found in model"):
            panda_robot.get_frame_pose("not_a_real_frame", panda_robot.config.home_position)


class TestJointPositionsToQ:
    def test_empty_returns_neutral(self, panda_robot):
        q = panda_robot.joint_positions_to_q({})
        np.testing.assert_allclose(q, pin.neutral(panda_robot.model))

    def test_revolute_joint_sets_position(self, panda_robot):
        """Panda's joints are JointModelRZ (nq=1) — value is written directly to idx_q."""
        pos_a = 0.5
        pos_b = -0.3
        q = panda_robot.joint_positions_to_q({0: pos_a, 2: pos_b})
        assert q[panda_robot.model.joints[1].idx_q] == pytest.approx(pos_a)
        assert q[panda_robot.model.joints[3].idx_q] == pytest.approx(pos_b)
        # Untouched joints remain at neutral.
        neutral = pin.neutral(panda_robot.model)
        for idx in (1, 3, 4, 5, 6, 7):
            q_idx = panda_robot.model.joints[idx + 1].idx_q
            assert q[q_idx] == pytest.approx(neutral[q_idx])

    def test_continuous_joint_uses_cos_sin(self, mobile_robot):
        """elrobot_mobile wheels are RevoluteUnbounded — stored as [cos, sin]."""
        angle = 0.7
        # joint_idx=1 corresponds to model.joints[2], which is wheel_1.
        wheel_joint_idx = 1
        wheel_joint = mobile_robot.model.joints[wheel_joint_idx + 1]
        # Sanity check: this is one of pinocchio's unbounded revolute variants.
        assert "Unbounded" in wheel_joint.shortname()

        q = mobile_robot.joint_positions_to_q({wheel_joint_idx: angle})
        assert q[wheel_joint.idx_q] == pytest.approx(np.cos(angle))
        assert q[wheel_joint.idx_q + 1] == pytest.approx(np.sin(angle))

    def test_planar_root_left_at_neutral(self, mobile_robot):
        """The planar root has nq=4, nv=3 — neither code path applies, stays neutral."""
        root_position = 1.23
        # joint_idx=0 -> model.joints[1] = planar root.
        q = mobile_robot.joint_positions_to_q({0: root_position})
        neutral = pin.neutral(mobile_robot.model)
        root = mobile_robot.model.joints[1]
        for offset in range(root.nq):
            assert q[root.idx_q + offset] == pytest.approx(neutral[root.idx_q + offset])

    def test_returns_correct_length(self, panda_robot):
        q = panda_robot.joint_positions_to_q({0: 0.1})
        assert q.shape == (panda_robot.model.nq,)


class TestJointVelocitiesToV:
    def test_empty_returns_zeros(self, panda_robot):
        v = panda_robot.joint_velocities_to_v({})
        np.testing.assert_array_equal(v, np.zeros(panda_robot.model.nv))

    def test_velocity_written_at_idx_v(self, panda_robot):
        vel_a = 1.5
        vel_b = -0.25
        v = panda_robot.joint_velocities_to_v({0: vel_a, 2: vel_b})
        assert v[panda_robot.model.joints[1].idx_v] == pytest.approx(vel_a)
        assert v[panda_robot.model.joints[3].idx_v] == pytest.approx(vel_b)

    def test_planar_root_velocity_ignored(self, mobile_robot):
        """The planar root has nv=3 — it should be skipped (left at zero)."""
        root_velocity = 9.9
        v = mobile_robot.joint_velocities_to_v({0: root_velocity})
        root = mobile_robot.model.joints[1]
        for offset in range(root.nv):
            assert v[root.idx_v + offset] == 0.0

    def test_continuous_joint_velocity_written(self, mobile_robot):
        """Continuous joints have nv=1 — velocity is written directly."""
        wheel_velocity = 2.5
        v = mobile_robot.joint_velocities_to_v({1: wheel_velocity})
        assert v[mobile_robot.model.joints[2].idx_v] == pytest.approx(wheel_velocity)

    def test_returns_correct_length(self, panda_robot):
        v = panda_robot.joint_velocities_to_v({0: 0.0})
        assert v.shape == (panda_robot.model.nv,)


class TestJointIdxLookup:
    def test_position_idx_matches_model(self, panda_robot):
        for joint_idx in range(panda_robot.model.njoints - 1):
            expected = panda_robot.model.joints[joint_idx + 1].idx_q
            assert panda_robot.joint_idx_to_position_idx(joint_idx) == expected

    def test_velocity_idx_matches_model(self, panda_robot):
        for joint_idx in range(panda_robot.model.njoints - 1):
            expected = panda_robot.model.joints[joint_idx + 1].idx_v
            assert panda_robot.joint_idx_to_velocity_idx(joint_idx) == expected

    def test_skips_universe_joint(self, mobile_robot):
        """joint_idx=0 maps to model.joints[1] (root), not the universe joint."""
        assert mobile_robot.joint_idx_to_position_idx(0) == mobile_robot.model.joints[1].idx_q
        assert mobile_robot.joint_idx_to_velocity_idx(0) == mobile_robot.model.joints[1].idx_v


class TestSetGripperPositions:
    def test_writes_at_position_index_for_fixed_base(self, panda_robot):
        """Panda's gripper joint_idx happens to equal its position_idx (7)."""
        gripper_joint_idx = panda_robot.config.gripper_joint_indices[0]
        position_idx = panda_robot.joint_idx_to_position_idx(gripper_joint_idx)

        q = pin.neutral(panda_robot.model)
        value = 0.0321
        panda_robot.set_gripper_positions(q, np.array([value]))

        assert q[position_idx] == pytest.approx(value)

    def test_writes_at_position_index_not_joint_index_on_mobile(self, mobile_robot):
        """Regression: the planar base shifts q, so position_idx != joint_idx (11 vs 17)."""
        joint_idx = mobile_robot.config.gripper_joint_indices[0]
        position_idx = mobile_robot.joint_idx_to_position_idx(joint_idx)
        assert joint_idx != position_idx, (
            "Test premise: mobile robot's gripper joint_idx must differ from position_idx."
        )

        q = pin.neutral(mobile_robot.model)
        decoy = 0.987
        value = 0.123
        q[joint_idx] = decoy  # value at the "wrong" slot we should NOT overwrite
        mobile_robot.set_gripper_positions(q, np.array([value]))

        assert q[position_idx] == pytest.approx(value)
        assert q[joint_idx] == pytest.approx(decoy), (
            "set_gripper_positions must not touch the joint_idx slot when it differs "
            "from position_idx — that was the original bug."
        )

    def test_no_grippers_configured_is_noop(self, panda_robot):
        no_gripper_config = replace(panda_robot.config, gripper_joint_indices=None)
        robot = Robot.__new__(Robot)
        robot.__dict__.update(panda_robot.__dict__)
        robot._config = no_gripper_config

        q = pin.neutral(robot.model)
        q_before = q.copy()
        # Passing a value despite no gripper joints should silently do nothing.
        robot.set_gripper_positions(q, np.array([99.0]))

        np.testing.assert_array_equal(q, q_before)

    def test_empty_gripper_list_is_noop(self, panda_robot):
        empty = replace(panda_robot.config, gripper_joint_indices=[])
        robot = Robot.__new__(Robot)
        robot.__dict__.update(panda_robot.__dict__)
        robot._config = empty

        q = pin.neutral(robot.model)
        q_before = q.copy()
        robot.set_gripper_positions(q, np.array([]))

        np.testing.assert_array_equal(q, q_before)

    def test_wrong_count_raises_assertion(self, panda_robot):
        """gripper_positions length must match the number of configured gripper joints."""
        q = pin.neutral(panda_robot.model)
        with pytest.raises(AssertionError, match="invalid number of gripper_positions"):
            panda_robot.set_gripper_positions(q, np.array([0.1, 0.2]))

    def test_mutates_in_place(self, panda_robot):
        """The function returns None and mutates the supplied q."""
        gripper_joint_idx = panda_robot.config.gripper_joint_indices[0]
        position_idx = panda_robot.joint_idx_to_position_idx(gripper_joint_idx)

        q = pin.neutral(panda_robot.model)
        q_id = id(q)
        result = panda_robot.set_gripper_positions(q, np.array([0.05]))

        assert result is None
        assert id(q) == q_id
        assert q[position_idx] == pytest.approx(0.05)


def test_print_info_runs(panda_robot, capsys):
    """print_info shouldn't raise and should write something."""
    panda_robot.print_info()
    captured = capsys.readouterr()
    assert "nq" in captured.out
    assert "nv" in captured.out
