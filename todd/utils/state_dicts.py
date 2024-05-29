__all__ = [
    'StateDict',
    'StateDictMixin',
    'transfer_weight',
    'transfer_weights',
]

from typing import Any, Mapping, TypeVar

import torch
from torch import nn

from ..loggers import logger
from ..patches import get_, get_rank

T = TypeVar('T')

StateDict = dict[str, torch.Tensor]


class StateDictMixin:

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        pass


# TODO: rename and support StateDictMixin


def transfer_weight(target: nn.Module, source: nn.Module) -> None:
    state_dict = source.state_dict()
    incompatible_keys = target.load_state_dict(state_dict, strict=False)
    if get_rank() == 0:
        logger.info(incompatible_keys)


def transfer_weights(models, weight_prefixes: Mapping[str, str]) -> None:
    for target_prefix, source_prefix in weight_prefixes.items():
        target = get_(models, target_prefix)
        source = get_(models, source_prefix)
        transfer_weight(target, source)
