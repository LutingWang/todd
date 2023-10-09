__all__ = [
    'transfer_weight',
    'transfer_weights',
]

from typing import Mapping

import torch.nn as nn

from ..utils import get_
from .logger import logger


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


def transfer_weights(models, weight_prefixes: Mapping[str, str]) -> None:
    for target_prefix, source_prefix in weight_prefixes.items():
        target = get_(models, target_prefix)
        source = get_(models, source_prefix)
        transfer_weight(target, source)
