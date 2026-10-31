# Cleanup task

The simulation-only cleanup task fetches loose toys with Triskel and deposits them
in a bin. `task.py` implements `CleanupTask`; `scene.py`, `motion.py`, `kinematics.py`,
`navigation.py`, and `visibility.py` contain its scene and expert helpers.

Concrete defaults live in [config/tasks/cleanup.py](../../../config/tasks/cleanup.py).
Configuration and planning shapes live in
[types/tasks/cleanup.py](../../../types/tasks/cleanup.py).
See the [shared simulation documentation](../../README.md) for the task interface,
collector CLI, LCM recording contract, and verification commands.

## Expert and scene

Each episode contains three blocks and a rigid toy car at four distant locations,
with seeded position jitter of ±6 mm and yaw jitter of ±0.3 radians. This is bounded
layout variation around a known, reachable scene, not arbitrary clutter generation.
The bin remains fixed. All objects move through contact physics after initialization;
there are no attachment welds, object teleports, or kinematic object carrying.

The expert queries current body poses and bounding boxes that include compound
collision geometry (including car wheels). It projects the oriented boxes onto the
floor and inflates them by a 20 cm circular swept robot footprint plus 15 mm clearance.
A visibility graph finds the shortest translation route in this conservative polygon
map. Pickup endpoints lie 30.5 cm from the toy at its four box-face grasp headings.
This keeps the gripper roll aligned with the camera and avoids unstable corner grasps.
The planner chooses the shortest feasible pickup route among these endpoints. For the
return to the bin it jointly optimizes over the continuous standoff circle.
The 32-sided inflation overestimates the disk by at most 1.04 mm. Direct clear routes
are straight lines, including the return to the nearest reachable side of the bin.
This optimizes each leg; collection order selects the nearest remaining toy from current scene poses.

The base turns in place to face the destination, tracks its bearing while translating,
and settles within the position/yaw tolerances before manipulation. Triskel's front
is its local +Y direction, calibrated by `pickup_offset`. Translation and rotation
have separate velocity/acceleration limits; the expert checks the measured base pose.
During collection, motion references pass through the same orchestrator and robot
controller as teleoperation. The shared controller generates the actual joint and wheel
commands by integrating its commanded configuration open-loop, as in teleoperation.
The expert's task checks still use measured scene state. Grasp transfers verify
the reached tool pose, allowing the redundant arm to choose different joint angles.
The empty arm retracts to its home pose before travel; while carrying, it preserves
the lifted grasp pose. The carried object is excluded from obstacles. The expert positions the base,
preflights its arm trajectory, closes the fingers, verifies a lift, and returns to
the leftmost empty drop slot in the current bin view. Occupancy comes from the current
scene rather than an implicit pickup index. Placement compensates the measured
object-in-gripper transform and levels the object before release. Pose and gripper checks detect lost
objects. Failed grasps are retried within a bounded attempt count; motion/time-budget
failures produce unsuccessful episodes.

Success requires **every OBB corner** to lie inside the bin interior with 2 mm contact
tolerance, below its rim, and above its floor; linear and angular speeds must be small.
The condition must persist over multiple checks after settling. The collector checks
success again independently before labeling the episode successful.

The current physical model uses an ideal planar mobile base and disables robot
self-collision. The arm motions assume this clear, static layout. These demonstrations
are not evidence of general collision avoidance or real-hardware performance.

## Four RGB cameras

The camera is mounted on `arm_6_link`, on the two small mounting holes behind the
gripper. Hole positions come from the last-link mesh: 3.25 mm bores spaced 20.5 mm
apart. A simplified bracket and screws support the visible camera housing. The camera
moves with the last arm link and stays fixed when the downstream gripper roll joint
(`arm_7`) rotates. The configurable optical transform remains a proposed hand-eye
calibration. The simulated RGB camera uses 848×480 images and an 87°×58° FOV.

Three additional cameras attach to `lower_plate_link`, one between each adjacent pair of
wheels. Their names are `base_1_2`, `base_2_3`, and `base_3_1`; their outward headings
are −30°, 90°, and 210° in the base frame. Lens positions are 108 mm from the base
center and 29 mm above its origin, pointing horizontally outward. Their brackets sit on the
bottom plate at z=7 mm. The complete housings fit below the upper plate at z=53 mm.
Each records 1920×1080 RGB with a configurable 150°×90° FOV. The wide horizontal views
overlap so destinations can cross between cameras during turns. The higher resolution
provides useful detail for distant toys. These are proposed
simulation mounting and optical parameters, pending hardware selection/calibration.
Each housing adds mass and inertia to the robot and appears in scene replay.

MuJoCo renders RGB, encoded as JPEG for LCM recording. No depth pass, depth noise model,
or depth payload is used. Intrinsics and privileged camera extrinsics accompany each
frame. The expert plans from privileged object poses and bounding boxes independently
of the images; RGB is the policy's camera input.

Version 9 moves the base cameras to the bottom plate and removes mandatory target
visibility, camera-aiming arm postures, wrist tracking, and camera-search turns. Acquisition uses the nearest
remaining target. Visibility and coverage remain optional diagnostics in labels.
All four cameras still record every sample. Actuator targets remain unfiltered;
matching real actuator dynamics is deferred to system identification.

Gripper motor torque, friction, contact solver, and coupling solver settings are
owned by `RobotConfig.gripper.simulation`. The provisional profile is defined in
[config/robot/triskel.py](../../../config/robot/triskel.py) and recorded in episode
metadata. Its sliding friction coefficient is 1.0. Controlled lifts showed that
doubling that coefficient did not fix dynamic slip: implicit integration caused
millimeters of displacement despite sufficient friction force. Cleanup now uses
Euler at 0.25 ms, retaining elliptic cones, friction impedance ratio 100 and 20 NoSlip
iterations. The regression lifts each toy to full transit height, holds for three
seconds, turns the base, and allows at most 1 mm of relative displacement.
Comparisons at 0.5 ms and with RK4 produced comparable submillimeter slip;
0.25 ms also resolves the fast base yaw servo without transient overshoot. These are
numerical checks, not a calibration to measured hardware behavior.

## Training observations

The intended actor receives **four current RGB frames**, with no depth, history,
object poses, phase labels, or camera/world extrinsics. The expert still uses privileged
state to plan; visibility checks are simulation QA, not a perception stack.

Target visibility does not gate acquisition, motion, success, or recording eligibility.
The expert can continue while a target is occluded. Optional diagnostics project
calibrated OBBs and cast rays against rendered geometry, including the robot and held
objects. The `views` label contains the selected camera, pixel area, visible fraction,
and framing measurements; `visible: false` is a valid observation. Segmentation is
used only in tests, never recorded as actor input.

Every sample must contain all four distinct cameras declared in the episode metadata.
The collector and raw-log auditor still reject missing or corrupt RGB, invalid
boundaries, timestamp errors, and missing samples. Physical success still requires
all toys to be deposited and settled in the bin.

Four cameras provide broad coverage but do not guarantee that every expert decision
can be inferred from one image set. Route planning uses privileged state, and absolute
joint commands can depend on configurations not uniquely determined by RGB. The future
converter must choose an appropriate relative action representation and control
interval without feeding privileged labels to the actor. Conversion, training, broader
randomization, and evaluation on held-out layouts remain subsequent work.
