__all__ = [
    'XAnchor',
    'YAnchor',
]

import enum


class XAnchor(enum.Enum):
    LEFT = enum.auto()
    RIGHT = enum.auto()


class YAnchor(enum.Enum):
    TOP = enum.auto()
    BOTTOM = enum.auto()
