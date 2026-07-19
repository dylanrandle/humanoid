"""Configuration types for root-state estimators."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RootStateEstimatorConfig:
    """Base configuration for one root-state estimator implementation."""
