__all__ = [
    'DistillerRegistry',
    'PipelineRegistry',
]

from ..registries import KnowledgeDistillationRegistry


class DistillerRegistry(KnowledgeDistillationRegistry):
    pass


class PipelineRegistry(KnowledgeDistillationRegistry):
    pass
