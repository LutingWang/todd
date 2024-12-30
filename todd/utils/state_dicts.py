__all__ = [
    'StateDict',
    'Keys',
    'StateDictMixin',
    'transfer_state_dict',
    'transfer_state_dicts',
    'StateDictConverter',
]

import functools
import re
from collections import defaultdict
from itertools import starmap
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


class UnknownKeyError(ValueError):

    def __init__(self, key: str, *args) -> None:
        super().__init__(f"'{key}'", *args)


class StateDictConverter:

    @staticmethod
    def _remove_prefix(key: str, prefix: str) -> str:
        assert key.startswith(prefix)
        return key.removeprefix(prefix)

    def __init__(
        self,
        *args,
        module: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._module = module
        self._key_mapping: dict[str, str | None] = dict()
        self._child_converters: dict[
            str,
            tuple[str | None, type[StateDictConverter]],
        ] = dict()
        self._regex_converters: dict[re.Pattern[str], str | None] = dict()

    @property
    def module(self) -> nn.Module:
        assert self._module is not None
        return self._module

    def _register_key_mapping(self, key: str, new_key: str | None) -> None:
        self._key_mapping[key] = new_key

    def _register_child_converter(
        self,
        child_name: str,
        new_child_name: str | None,
        child_converter_type: type['StateDictConverter'],
    ) -> None:
        self._child_converters[child_name] = (
            new_child_name,
            child_converter_type,
        )

    def _register_regex_converter(
        self,
        key_pattern: str | re.Pattern[str],
        new_key: str | None,
    ) -> None:
        if isinstance(key_pattern, str):
            key_pattern = re.compile(key_pattern)
        self._regex_converters[key_pattern] = new_key

    def _child_converter(
        self,
        child_name: str,
    ) -> tuple[str | None, 'StateDictConverter']:
        new_child_name, child_converter_type = (
            self._child_converters[child_name]
        )
        if self._module is None:
            child = None
        elif new_child_name is None:
            child = self._module
        else:
            child = getattr(self._module, new_child_name)
        return new_child_name, child_converter_type(module=child)

    def load(self, *args, **kwargs) -> StateDict:
        return load_state_dict_(*args, **kwargs)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        return state_dict

    def _convert(self, key: str) -> str | None:
        raise UnknownKeyError(key)

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        return state_dict

    def _convert_child_state_dict(
        self,
        name: str,
        state_dict: StateDict,
    ) -> StateDict:
        new_child_name, child_converter = self._child_converter(name)
        prefix = '' if new_child_name is None else f'{new_child_name}.'
        state_dict = child_converter.convert(state_dict)
        return {prefix + k: v for k, v in state_dict.items()}

    def convert(self, state_dict: StateDict) -> StateDict:  # noqa: C901
        state_dict = self._pre_convert(state_dict)

        new_state_dict: StateDict = dict()

        for key, new_key in self._key_mapping.items():
            if key not in state_dict:
                continue
            value = state_dict.pop(key)
            if new_key is not None:
                new_state_dict[new_key] = value

        # group child state dicts by child name
        child_state_dicts: defaultdict[str, StateDict] = defaultdict(dict)

        for key, value in state_dict.items():

            # store child state dicts for later conversion
            if self._child_converters and '.' in key:
                child_name, child_key = key.split('.', 1)
                if child_name in self._child_converters:
                    child_state_dicts[child_name][child_key] = value
                    continue

            # convert keys using regex
            for key_pattern, new_key in self._regex_converters.items():
                match_ = key_pattern.fullmatch(key)
                if match_ is None:
                    continue
                if new_key is not None:
                    new_state_dict[match_.expand(new_key)] = value
                break

            # convert keys using `_convert`
            else:
                new_key = self._convert(key)
                if new_key is not None:
                    new_state_dict[new_key] = value

        # convert child state dicts
        for child_state_dict in starmap(
            self._convert_child_state_dict,
            child_state_dicts.items(),
        ):
            new_state_dict.update(child_state_dict)

        new_state_dict = self._post_convert(new_state_dict)
        return new_state_dict


def parallel_conversion(func):

    @functools.wraps(func)
    def wrapper(self: StateDictConverter, state_dict: StateDict) -> StateDict:
        state_dicts: defaultdict[str, StateDict] = defaultdict(dict)
        for key, value in state_dict.items():
            prefix, key = key.split('.', 1)
            state_dicts[prefix][key] = value
        for prefix in state_dicts:
            state_dicts[prefix] = func(self, state_dicts[prefix], prefix)
        state_dict = {
            f'{prefix}.{k}': v
            for prefix, state_dict in state_dicts.items()
            for k, v in state_dict.items()
        }
        return state_dict

    return wrapper
