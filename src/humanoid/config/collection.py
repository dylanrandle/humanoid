"""Defaults for autonomous simulation data collection."""

from humanoid.types.collection import CollectionConfig, CollectionSettings

COLLECTION_SCHEMA_VERSION = 4
DEFAULT_COLLECTION_CONFIG = CollectionConfig(startup_timeout=15.0)
DEFAULT_COLLECTION_SETTINGS = CollectionSettings(
    task="empty",
    episodes=1,
    max_attempts=3,
    seed=0,
    max_sim_seconds=DEFAULT_COLLECTION_CONFIG.max_sim_seconds,
    output="logs/recordings",
)
