"""Manufacturer effort ratings and current calibration for physical actuators."""

from humanoid.types.actuator import ActuatorCurrentCalibration, ActuatorEffortLimits

KGF_CM_TO_NM = 0.0980665

# HLS3930M-C001 datasheet section 5-11: output-shaft Kt, not motor-side Kt.
# https://www.feetechrc.com/Data/feetechrc/upload/file/20240823/6386000476147063735440942.pdf
HLS3930M_CURRENT_CALIBRATION = ActuatorCurrentCalibration(
    amperes_per_unit=0.0065,
    effort_per_ampere=12.5 * KGF_CM_TO_NM,
)
HLS3930M_EFFORT_LIMITS = ActuatorEffortLimits(
    stall=35.0 * KGF_CM_TO_NM,
    rated=8.7 * KGF_CM_TO_NM,
)

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
