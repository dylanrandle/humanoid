# State estimation

State estimation turns measurements into robot feedback. It is separate from
hardware access and robot kinematics.

## Base odometry

Triskel uses wheel dead reckoning in simulation and on hardware. Robot-owned
wheel kinematics convert measured wheel motion into planar velocity; the estimator
integrates that velocity to estimate base pose. There is no absolute base-position
measurement, so this estimate can drift.

Future sensor fusion belongs in the estimation package, consuming raw measurements
from hardware interfaces and model operations from `robots`. Estimators do not own
device connection lifecycles.

## See also

- [Documentation index](README.md)
- [Architecture](architecture.md)
- [Root estimators](../src/humanoid/state_estimation/root/)
- [Wheel kinematics](../src/humanoid/robots/wheels.py)
