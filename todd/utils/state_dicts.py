__all__ = [
    'StateDict',
    'Keys',
    'StateDictMixin',
    'transfer_state_dict',
    'transfer_state_dicts',
    'StateDictConverter',
]

from abc import ABC, abstractmethod
from typing import Any, Mapping, NamedTuple, TypeVar

import torch
from torch import nn

from ..loggers import master_logger
from ..patches.py_ import get_
from ..patches.torch import load_state_dict_

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
    target: nn.Module | StateDictMixin,
    source: nn.Module | StateDictMixin,
) -> None:
    state_dict = source.state_dict()
    keys = target.load_state_dict(state_dict, strict=False)
    if keys is not None:
        missing, unexpected = keys
        master_logger.info(
            "\nMissing keys: %s\nUnexpected keys: %s",
            ', '.join(missing),
            ', '.join(unexpected),
        )


def transfer_state_dicts(models: Any, prefixes: Mapping[str, str]) -> None:
    for target_prefix, source_prefix in prefixes.items():
        target = get_(models, target_prefix)
        source = get_(models, source_prefix)
        transfer_state_dict(target, source)


class StateDictConverter(ABC):

    def load(self, *args, **kwargs) -> StateDict:
        return load_state_dict_(*args, **kwargs)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        return state_dict

    @abstractmethod
    def _convert(self, key: str) -> str | None:
        pass

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        return state_dict

    def convert(self, state_dict: StateDict) -> StateDict:
        state_dict = self._pre_convert(state_dict)
        state_dict = {
            converted_k: v
            for k, v in state_dict.items()
            if (converted_k := self._convert(k)) is not None
        }
        state_dict = self._post_convert(state_dict)
        return state_dict
