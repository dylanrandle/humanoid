"""Task labels shared by simulated experts and real demonstration producers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskAnnotation:
    """Task-specific labels at a time on the episode's collection clock.

    labels_json is a JSON object whose schema belongs to task/task_version, not
    to a simulator. Labels are supervision, not automatically policy inputs.
    """

    timestamp: float
    episode_id: str
    task: str
    task_version: int
    labels_json: str
