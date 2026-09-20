"""Manufacturer effort ratings and current calibration for physical actuators."""

from humanoid.types.actuator import ActuatorCurrentCalibration, ActuatorEffortLimits

KGF_CM_TO_NM = 0.0980665

# Feetech ST-3215-C018, 12 V, specification dated 2023-07-20:
# https://akizukidenshi.com/goodsaffix/STS3215-C018.pdf (section 5).
STS3215_12V_EFFORT_LIMITS = ActuatorEffortLimits(
    stall=30.0 * KGF_CM_TO_NM,
    rated=10.0 * KGF_CM_TO_NM,
)
# Current feedback is 6.5 mA/unit, not the PWM-based load register:
# https://www.feetech.cn/en/2020-05-13_56655.html (section 8).
STS3215_12V_CURRENT_CALIBRATION = ActuatorCurrentCalibration(
    amperes_per_unit=0.0065,
    effort_per_ampere=11.0 * KGF_CM_TO_NM,
)
