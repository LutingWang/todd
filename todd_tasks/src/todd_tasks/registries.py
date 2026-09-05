__all__ = [
    'TaskRegistry',
    'IGRegistry',
    'KDRegistry',
    'ODRegistry',
    'ODKDRegistry',
    'OFERegistry',
    'PTRegistry',
    'LMMRegistry',
    'NLPRegistry',
]

from todd.bases.registries import Registry


class TaskRegistry(Registry):
    pass


class IGRegistry(TaskRegistry):
    pass


class KDRegistry(TaskRegistry):
    pass


class ODRegistry(TaskRegistry):
    pass


class ODKDRegistry(TaskRegistry):
    pass


class OFERegistry(TaskRegistry):
    pass


class PTRegistry(TaskRegistry):
    pass


class LMMRegistry(TaskRegistry):
    pass


class NLPRegistry(TaskRegistry):
    pass
