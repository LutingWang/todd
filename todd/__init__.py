"""Toolkit for Object Detection Distillation."""
from . import _patches  # noqa: F401
from . import (
    adapts,
    distillers,
    hooks,
    losses,
    reproduction,
    schedulers,
    utils,
    visuals,
)
from ._base import *  # noqa: F401,F403
from ._patches import *  # noqa: F401,F403

__all__ = [
    'adapts',
    'distillers',
    'hooks',
    'losses',
    'reproduction',
    'schedulers',
    'utils',
    'visuals',
]
__version__ = '0.0.1'
__author__ = 'Luting Wang'
