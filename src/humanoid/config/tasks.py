"""Provisional task geometry; car height, mass and bin width still need measurement."""

from humanoid.types.tasks import ToyCarCleanupConfig

TOY_CAR_CLEANUP_CONFIG = ToyCarCleanupConfig(
    room_min=(-0.8, -0.4),
    room_max=(0.8, 1.6),
    ground_z=-0.032,
    car_count=3,
    car_length=0.050,
    car_width=0.025,
    car_height=0.022,
    car_mass=0.040,
    bin_center=(-0.48, 0.40),
    bin_inner_size=(0.24, 0.18),
    bin_rim_height=0.127,  # Five inches above the floor.
    bin_wall_thickness=0.006,
    settle_seconds=1.0,
)
