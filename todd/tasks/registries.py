__all__ = [
    'ImageGenerationRegistry',
    'KnowledgeDistillationRegistry',
    'ObjectDetectionRegistry',
    'OpticalFlowRegistry',
]

from ..registries import TaskRegistry


class ImageGenerationRegistry(TaskRegistry):
    pass


class KnowledgeDistillationRegistry(TaskRegistry):
    pass


class ObjectDetectionRegistry(TaskRegistry):
    pass


class OpticalFlowRegistry(TaskRegistry):
    pass
