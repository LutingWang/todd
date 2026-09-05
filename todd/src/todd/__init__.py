"""Core utilities for Toolkit for Object Detection Distillation."""

__version__ = '0.7.1'

from . import colors, configs, loggers, patches, registries, utils
from .configs import Config
from .loggers import logger
from .patches import *
from .registries import Registry, RegistryMeta
from .utils import Store
