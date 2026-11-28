"""Native worker limits for the parallel verification suite."""

# Pytest and the simulation stacks already supply process-level parallelism.
# Avoid another machine-sized worker pool in every numerical library and camera.
VERIFICATION_ENVIRONMENT: dict[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "LP_NUM_THREADS": "1",
}
