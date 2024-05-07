__all__ = [
    'InitRegistry',
]

from torch.nn import init

from .partials import PartialRegistry


class InitRegistry(PartialRegistry):
    pass


InitRegistry.register_()(init.uniform_)
InitRegistry.register_()(init.normal_)
InitRegistry.register_()(init.trunc_normal_)
InitRegistry.register_()(init.constant_)
InitRegistry.register_()(init.ones_)
InitRegistry.register_()(init.zeros_)
InitRegistry.register_()(init.eye_)
InitRegistry.register_()(init.dirac_)
InitRegistry.register_()(init.xavier_uniform_)
InitRegistry.register_()(init.xavier_normal_)
InitRegistry.register_()(init.kaiming_uniform_)
InitRegistry.register_()(init.kaiming_normal_)
InitRegistry.register_()(init.orthogonal_)
InitRegistry.register_()(init.sparse_)
