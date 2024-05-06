"""Control sequences as per ECMA-48_.

.. _ECMA-48: https://ecma-international.org/publications-and-standards/standards/ecma-48/  # noqa: E501 pylint: disable=line-too-long
"""

__all__ = [
    'CSI',
    'control_sequence',
    'SGR',
    'sgr',
    'apply_sgr',
]

import enum
import itertools
from typing import Any, Iterable

CSI = '\033['


def control_sequence(
    parameter_bytes: Iterable[Any],
    intermediate_bytes: Iterable[Any],
    final_byte: str,
) -> str:
    return (
        CSI + ';'.join(map(str, parameter_bytes))
        + ''.join(map(str, intermediate_bytes)) + final_byte
    )


class SGR(enum.IntEnum):
    """Part of the SGR values."""

    DEFAULT = 0
    BOLD = enum.auto()
    FAINT = enum.auto()
    ITALICIZED = enum.auto()
    SINGLY_UNDERLINED = enum.auto()
    SLOWLY_BLINKING = enum.auto()
    RAPIDLY_BLINKING = enum.auto()
    NEGATIVE_IMAGE = enum.auto()
    CONCEALED_CHARACTERS = enum.auto()
    CROSSED_OUT = enum.auto()

    DISPLAY_BLACK = 30
    DISPLAY_RED = enum.auto()
    DISPLAY_GREEN = enum.auto()
    DISPLAY_YELLOW = enum.auto()
    DISPLAY_BLUE = enum.auto()
    DISPLAY_MAGENTA = enum.auto()
    DISPLAY_CYAN = enum.auto()
    DISPLAY_WHITE = enum.auto()

    BACKGROUND_BLACK = 40
    BACKGROUND_RED = enum.auto()
    BACKGROUND_GREEN = enum.auto()
    BACKGROUND_YELLOW = enum.auto()
    BACKGROUND_BLUE = enum.auto()
    BACKGROUND_MAGENTA = enum.auto()
    BACKGROUND_CYAN = enum.auto()
    BACKGROUND_WHITE = enum.auto()


def sgr(parameter_bytes: Iterable[SGR]) -> str:
    return control_sequence([pb.value for pb in parameter_bytes], [], 'm')


def apply_sgr(str_: str, *parameter_bytes: SGR) -> str:
    return sgr(parameter_bytes) + str_ + sgr([])


def sgr_cli() -> None:
    for fg, bg in itertools.product(range(30, 38), range(40, 48)):
        text = ' '.join(
            apply_sgr(f'{effect}:{fg};{bg}', SGR(effect), SGR(fg), SGR(bg))
            for effect in range(10)
        )
        print(text)
