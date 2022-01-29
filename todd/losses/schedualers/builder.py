import numbers
from typing import Union

from mmcv.utils import Registry

from .base import BaseSchedualer


SCHEDUALERS = Registry('schedualers')

SchedualerConfig = Union[BaseSchedualer, numbers.Number, dict]


def build_schedualer(cfg: SchedualerConfig) -> BaseSchedualer:

    from .constant import ConstantSchedualer

    if isinstance(cfg, BaseSchedualer):
        return cfg
    if isinstance(cfg, numbers.Number):
        return ConstantSchedualer(cfg)
    assert isinstance(cfg, dict)
    return SCHEDUALERS.build(cfg)
