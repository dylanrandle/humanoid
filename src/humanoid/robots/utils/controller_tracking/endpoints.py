"""Robot-specific endpoints for controller-tracking comparisons."""

from humanoid.types.controller_tracking import (
    ControllerTrackingComparisonConfig,
    ControllerTrackingEndpoint,
)
from humanoid.types.robot import RobotName

TRISKEL_CONTROLLER_TRACKING_COMPARISON = ControllerTrackingComparisonConfig(
    start=ControllerTrackingEndpoint(
        joint_positions_rad={
            "arm_1": -0.8638443724775258,
            "arm_2": -0.263732934633247,
            "arm_3": 0.6843106472653648,
            "arm_4": 0.11493303000021075,
            "arm_5": -0.9901418694975344,
            "arm_6": 0.7035938721391839,
            "arm_7": 1.537319663155239,
            "gripper_1": 1.9470564256610957e-08,
        },
        task_frame="root_joint",
        task_position_m=(
            0.1012100874014162,
            0.2042972651386029,
            0.1548551919777746,
        ),
        task_quaternion_wxyz=(
            0.9300895287166265,
            -0.367147216514987,
            0.0064482733938382875,
            0.009737029691811665,
        ),
    ),
    end=ControllerTrackingEndpoint(
        joint_positions_rad={
            "arm_1": 0.8346730263390332,
            "arm_2": -0.376690752501204,
            "arm_3": 0.6489220999615097,
            "arm_4": -0.4060530758400009,
            "arm_5": 2.580586314370916,
            "arm_6": 1.0434771694410234,
            "arm_7": -2.1466193161815688,
            "gripper_1": 4.067045836202966e-08,
        },
        task_frame="root_joint",
        task_position_m=(
            -0.09906640428017018,
            0.20433698521437949,
            0.32154807142970665,
        ),
        task_quaternion_wxyz=(
            0.8950104124102456,
            0.4459091579419254,
            0.01077157920641783,
            0.002314653789443016,
        ),
    ),
)

CONTROLLER_TRACKING_COMPARISONS: dict[
    RobotName,
    ControllerTrackingComparisonConfig,
] = {
    RobotName.TRISKEL: TRISKEL_CONTROLLER_TRACKING_COMPARISON,
}
