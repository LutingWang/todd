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
from .misc import set_temp

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
        assert '.' not in child_name
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

    def load(self, *args, **kwargs) -> StateDict:
        return load_state_dict_(*args, **kwargs)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        return state_dict

    def _map_keys(self, state_dict: StateDict) -> StateDict:
        new_state_dict: StateDict = dict()
        for key, new_key in self._key_mapping.items():
            if key not in state_dict:
                continue
            value = state_dict.pop(key)
            if new_key is not None:
                new_state_dict[new_key] = value
        return new_state_dict

    def _convert_child(self, name: str, state_dict: StateDict) -> StateDict:
        new_name, converter_type = self._child_converters[name]

        prefix = '' if new_name is None else f'{new_name}.'

        if self._module is None:
            module = None
        elif new_name is None:
            module = self._module
        else:
            module = getattr(self._module, new_name)
        converter = converter_type(module=module)

        new_state_dict = converter.convert(state_dict)
        return {prefix + k: v for k, v in new_state_dict.items()}

    def _convert_children(self, state_dict: StateDict) -> StateDict:
        new_state_dict: StateDict = dict()
        if not self._child_converters:
            return new_state_dict

        children: defaultdict[str, StateDict] = defaultdict(dict)
        for key in list(state_dict):
            if '.' not in key:
                continue
            child_name, child_key = key.split('.', 1)
            if child_name not in self._child_converters:
                continue
            value = state_dict.pop(key)
            children[child_name][child_key] = value

        for child in starmap(self._convert_child, children.items()):
            new_state_dict.update(child)

        return new_state_dict

    def _convert_regex(self, state_dict: StateDict) -> StateDict:
        new_state_dict: StateDict = dict()
        if not self._regex_converters:
            return new_state_dict

        for key in list(state_dict):
            for pattern, new_key in self._regex_converters.items():
                match_ = pattern.fullmatch(key)
                if match_ is None:
                    continue
                value = state_dict.pop(key)
                if new_key is not None:
                    new_key = match_.expand(new_key)
                    new_state_dict[new_key] = value
                break

        return new_state_dict

    def _convert(self, key: str) -> str | None:
        raise UnknownKeyError(key)

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        return state_dict

    def convert(self, state_dict: StateDict) -> StateDict:  # noqa: C901
        state_dict = self._pre_convert(state_dict)

        # move items from state_dict to new_state_dict
        new_state_dict = self._map_keys(state_dict)
        new_state_dict |= self._convert_children(state_dict)
        new_state_dict |= self._convert_regex(state_dict)

        for key, value in state_dict.items():
            new_key = self._convert(key)
            if new_key is not None:
                new_state_dict[new_key] = value

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


class SequentialStateDictConverterMixin(StateDictConverter):

    @parallel_conversion
    def convert(self, state_dict: StateDict, prefix: str) -> StateDict:  # noqa: E501 pylint: disable=arguments-differ
        module = self._module
        assert isinstance(module, nn.Sequential)
        with set_temp(self, '._module', module[int(prefix)]):
            return super().convert(state_dict)
