"""Toolkit for Object Detection Distillation."""

__version__ = '0.5.0'

from . import (
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
from .configs import Config
from .patches import *
from .registries import Registry
