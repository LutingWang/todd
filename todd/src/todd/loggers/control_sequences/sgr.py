__all__ = [
    'SGR',
    'sgr',
    'apply_sgr',
]

import enum
import itertools
from typing import Iterable

from .control_sequence import control_sequence


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
    r"""Apply SGR to a string.

    Args:
        str_: The string to apply SGR to.
        parameter_bytes: The SGR parameters.

    Returns:
        The string with the SGR applied.

    Examples:
        >>> apply_sgr(
        ...     ' hello ',
        ...     SGR.BOLD,
        ...     SGR.DISPLAY_GREEN,
        ...     SGR.BACKGROUND_GREEN,
        ... )
        '\x1b[1;32;42m hello \x1b[m'
    """
    return sgr(parameter_bytes) + str_ + sgr([])


def sgr_cli() -> None:
    for fg, bg in itertools.product(range(30, 38), range(40, 48)):
        text = ' '.join(
            apply_sgr(f'{effect}:{fg};{bg}', SGR(effect), SGR(fg), SGR(bg))
            for effect in range(10)
        )
        print(text)
