"""Manufacturer effort ratings and current calibration for physical actuators."""

from humanoid.types.actuator import ActuatorCurrentCalibration

KGF_CM_TO_NM = 0.0980665

# The M1 pendulum fit estimates apparent inertia at the 345:1 gearbox output.
# https://github.com/Rhoban/bam/blob/279c4828c466e7e19b631c45203b0e3582be9869/bam/params/feetech_sts3215_7_4V/m1.json
STS3215_345_OUTPUT_ARMATURE = 0.02608

# Section 5 of the C001 datasheet at 7.4 V. Stall torque is in the URDF.
# https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf
STS3215_7V4_RATED_TORQUE = 5.0 * KGF_CM_TO_NM
# https://www.feetech.cn/en/2020-05-13_56655.html (current feedback, section 8).
STS3215_7V4_CURRENT_CALIBRATION = ActuatorCurrentCalibration(
    amperes_per_unit=0.0065,
    effort_per_ampere=7.8 * KGF_CM_TO_NM,
)
