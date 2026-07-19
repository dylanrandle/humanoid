# State Estimation

State estimation is separate from hardware access and robot kinematics. This
package owns the domain state and algorithms used to produce robot state
feedback.

Planar root position uses Pinocchio's `[x, y, cos(yaw), sin(yaw)]`
representation. Velocity is the body-frame planar twist `[vx, vy, yaw_rate]`.

Elrobot Mobile uses `WheelDeadReckoningRootStateEstimator` in both simulation
and on real hardware. It composes the robot-owned `WheelKinematics` mapping with
the state-estimation-owned `DeadReckoningIntegrator`. Measured wheel positions
and velocities produce a body-frame planar velocity, which dead reckoning
integrates to estimate root pose. There is no absolute root-position measurement
yet.

Future sensor-fusion algorithms should be implemented here while consuming raw
measurements through the appropriate hardware interfaces and model operations
from `robots`. Estimation algorithms do not own hardware-style `connect()` or
`disconnect()` lifecycle methods.
