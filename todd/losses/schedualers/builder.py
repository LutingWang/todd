import numbers
from typing import Union

from mmcv.utils import Registry

from .base import BaseSchedualer


SCHEDUALERS = Registry('schedualers')

SchedualerCfg = Union[BaseSchedualer, numbers.Number, dict]


def build_schedualer(cfg: SchedualerCfg) -> BaseSchedualer:
    if isinstance(cfg, BaseSchedualer):
        return cfg
    if isinstance(cfg, numbers.Number):
        cfg = dict(type='ConstantSchedualer', value=cfg)
    return SCHEDUALERS.build(cfg)
