"""Toolkit for Object Detection Distillation."""

__all__ = [
    'configs',
    'datasets',
    'loggers',
    'models',
    'registries',
    'runners',
    'tasks',
    'utils',
    'visuals',
    'Config',
    'Registry',
]
__version__ = '0.4.0'

from . import (
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
from .configs import Config
from .patches import *
from .registries import Registry
