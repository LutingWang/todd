__all__ = [
    # 'load_open_mmlab_models',
    'transfer_weight',
    'transfer_weights',
]

import torch.nn as nn

from ..utils import get_
from .logger import logger

# from .configs import Config
# from .registries import RegistryMeta
# def load_open_mmlab_models(
#     registry: RegistryMeta,
#     config: Config,
#     config_options: Config | None = None,  # TODO: rename overload
#     ckpt: str | None = None,
# ) -> Module:
#     model = (
#         registry.build(config)
#         if config_options is None else registry.build(config_options, config)
#     )
#     if ckpt is not None:
#         import mmcv.runner
#         mmcv.runner.load_checkpoint(model, ckpt, map_location='cpu')
#         model._is_init = True
#     return model


def transfer_weight(target: nn.Module, source: nn.Module) -> None:
    state_dict = source.state_dict()
    missing_keys, unexpected_keys = target.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing_keys:
        logger.warning('missing_keys:', missing_keys)
    if unexpected_keys:
        logger.warning('unexpected_keys:', unexpected_keys)


def transfer_weights(models, weight_prefixes: dict[str, str]) -> None:
    for target_prefix, source_prefix in weight_prefixes.items():
        target = get_(models, target_prefix)
        source = get_(models, source_prefix)
        transfer_weight(target, source)
