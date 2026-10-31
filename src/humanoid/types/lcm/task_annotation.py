"""Wire conversions for task labels from either simulation or real collection."""

from humanoid.types.lcm import task_annotation_t
from humanoid.types.task_annotation import TaskAnnotation


def task_annotation_to_lcm(annotation: TaskAnnotation) -> task_annotation_t:
    message = task_annotation_t()
    message.timestamp = round(annotation.timestamp * 1e9)
    message.episode_id, message.task = annotation.episode_id, annotation.task
    message.task_version = annotation.task_version
    message.labels_json = annotation.labels_json
    return message


def task_annotation_from_lcm(message: task_annotation_t) -> TaskAnnotation:
    return TaskAnnotation(
        message.timestamp / 1e9,
        message.episode_id,
        message.task,
        message.task_version,
        message.labels_json,
    )
