__all__ = [
    'ModuleList',
    'ModuleDict',
    'Sequential',
    'training_modules',
    'named_training_modules',
    'trainable_parameters',
    'named_trainable_parameters',
    'load_state_dict',
    'load_state_dict_',
]

import logging
from typing import Any, Generator, Mapping

import torch
from torch import nn

from .serialization import load


class ModuleList(nn.ModuleList):

    def forward(self, *args, **kwargs) -> list[nn.Module]:
        return [m(*args, **kwargs) for m in self]


class ModuleDict(nn.ModuleDict):

    def forward(self, *args, **kwargs) -> dict[str, nn.Module]:
        return {k: m(*args, **kwargs) for k, m in self.items()}


class Sequential(nn.Sequential):

    def __init__(self, *args, unpack_args: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._unpack_args = unpack_args

    def forward(self, *args, **kwargs) -> tuple[Any, ...]:
        if not self._unpack_args:
            args, = args
        for m in self:
            args = (
                m(*args, **kwargs) if self._unpack_args else m(args, **kwargs)
            )
        return args


def training_modules(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[nn.Module, None, None]:
    for m in module.modules(*args, **kwargs):
        if m.training:
            yield m


def named_training_modules(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[tuple[str, nn.Module], None, None]:
    for name, m in module.named_modules(*args, **kwargs):
        if m.training:
            yield name, m


def trainable_parameters(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[nn.Parameter, None, None]:
    for parameter in module.parameters(*args, **kwargs):
        if parameter.requires_grad:
            yield parameter


def named_trainable_parameters(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[tuple[str, nn.Parameter], None, None]:
    for name, parameter in module.named_parameters(*args, **kwargs):
        if parameter.requires_grad:
            yield name, parameter


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *args,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    if logger is None:
        from ...loggers import master_logger as logger
    assert logger is not None

    incompatible_keys = module.load_state_dict(state_dict, *args, **kwargs)
    logger.info(incompatible_keys)


def load_state_dict_(
    f: torch.serialization.FILE_LIKE | list[torch.serialization.FILE_LIKE],
    *args,
    logger: logging.Logger | None = None,
    **kwargs,
) -> dict[str, Any]:
    f_list = f if isinstance(f, list) else [f]
    if logger is None:
        from ...loggers import master_logger as logger
    assert logger is not None

    state_dict: dict[str, Any] = dict()
    for f_ in f_list:
        logger.info("Loading model from %s", f_)
        state_dict.update(load(f_, 'cpu', *args, **kwargs))
    return state_dict
