__all__ = [
    'ClipGradRegistry',
]

from torch.nn import utils

from .partials import PartialRegistry


class ClipGradRegistry(PartialRegistry):
    pass


ClipGradRegistry.register_()(utils.clip_grad_norm_)
ClipGradRegistry.register_()(utils.clip_grad_value_)
