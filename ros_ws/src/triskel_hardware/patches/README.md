# Pinned STS dependency patch

This patch applies to pinned upstream revision
`bd95bdef4e6d641c7ad5c8af534cae94681056c2`. `apply.bash` checks and applies
it idempotently before Docker or a native build compiles the driver.

`sts-proportional-velocity-units.patch` fixes the driver's mode-0 proportional
speed path. `proportional_vel_max` and the STS packet field use raw steps/s,
while each joint's `max_velocity` uses rad/s. The pinned implementation compares
those values directly, so a 1000-step/s cap is incorrectly reduced to roughly
5 steps/s by a 4.71 rad/s joint limit. The patch converts the joint limit to raw
steps/s before clipping the generated unsigned cap.
