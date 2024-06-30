"""Toolkit for Object Detection Distillation."""

__version__ = '0.5.1'

from . import (
    bases,
    colors,
    configs,
    datasets,
    loggers,
    models,
    registries,
    runners,
    tasks,
    utils,
    visuals,
)
from .bases.configs import Config
from .bases.registries import Registry, RegistryMeta
from .loggers import logger
from .patches import *
from .utils import Store
