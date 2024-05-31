__all__ = [
    'StateDict',
    'Keys',
    'StateDictMixin',
    'transfer_state_dict',
    'transfer_state_dicts',
]

from typing import Any, Mapping, NamedTuple, TypeVar

import torch
from torch import nn

from ..loggers import master_logger
from ..patches.py import get_

T = TypeVar('T')

StateDict = dict[str, torch.Tensor]


class Keys(NamedTuple):
    missing: list[str]
    unexpected: list[str]


class StateDictMixin:

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> Keys | None:
        pass


def transfer_state_dict(
    target: nn.Module | StateDictMixin, source: nn.Module | StateDictMixin
) -> None:
    state_dict = source.state_dict()
    keys = target.load_state_dict(state_dict, strict=False)
    if keys is not None:
        missing, unexpected = keys
        master_logger.info(
            "\nMissing keys: %s\nUnexpected keys: %s", ', '.join(missing),
            ', '.join(unexpected)
        )


def transfer_state_dicts(models: Any, prefixes: Mapping[str, str]) -> None:
    for target_prefix, source_prefix in prefixes.items():
        target = get_(models, target_prefix)
        source = get_(models, source_prefix)
        transfer_state_dict(target, source)
