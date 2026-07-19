"""Configuration types for root-state estimators."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RootStateEstimatorConfig:
    """Base configuration for one root-state estimator implementation.

    Subclasses must give ``kind`` a stable, implementation-specific default so
    recording snapshots distinguish estimators even when they have no settings.
    """

    kind: str
