import inspect

from humanoid.controllers.base import Controller
from humanoid.controllers.gripper import GripperController
from humanoid.controllers.omniwheel_base import OmniwheelBaseController
from humanoid.controllers.operational_space import OperationalSpaceController


def test_controller_defines_common_abstract_interface():
    assert inspect.isabstract(Controller)
    assert Controller.__abstractmethods__ == {"update_state", "compute_control"}


def test_all_control_strategies_implement_controller_interface():
    assert issubclass(OperationalSpaceController, Controller)
    assert issubclass(OmniwheelBaseController, Controller)
    assert issubclass(GripperController, Controller)
