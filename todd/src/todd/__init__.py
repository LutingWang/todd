"""Core utilities for Toolkit for Object Detection Distillation."""

__version__ = '0.7.1'

from . import bases, colors, configs, loggers, patches, registries, utils
from .bases.configs import Config
from .bases.registries import Registry, RegistryMeta
from .loggers import logger
from .patches import *
from .utils import Store
