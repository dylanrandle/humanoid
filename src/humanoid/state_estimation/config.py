"""Robot-level state-estimation configuration."""

from dataclasses import dataclass

from humanoid.state_estimation.root.config import RootStateEstimatorConfig


@dataclass(frozen=True, kw_only=True)
class RobotStateEstimationConfig:
    """State-estimation pipelines configured for one robot."""

    root: RootStateEstimatorConfig | None = None
