"""Operator-owned defaults for the local Linux RAX runner."""

RAX_IMAGE = "humanoid-rax:4b577aa2-codex-0.169.0"
RAX_MODEL = "gpt-6-astra"
# Leave room for iterative task learning and a separate qualifying rehearsal.
RAX_EXPLORATION_SEEDS = (0, 1, 2, 3, 4, 5)
RAX_FORMAL_SEED = 100
RAX_FORMAL_EPISODES = 3
RAX_LIMITS = {
    "wall_seconds": 1800,
    "memory_mb": 4096,
    "cpus": 2.0,
    "pids": 256,
    "scratch_mb": 1024,
    "artifact_bytes": 100 * 1024 * 1024,
}

RAX_CAMERA_TIMEOUT = 0.5
