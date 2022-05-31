from abc import abstractstaticmethod
from enum import IntEnum, auto
from typing import Iterable, Union


class ANSI:
    @abstractstaticmethod
    def format(text: str, *args, **kwargs) -> str:
        pass


class SGR(ANSI, IntEnum):
    """Select Graphic Rendition.

    Refer to https://en.wikipedia.org/wiki/ANSI_escape_code.

    """
    NORMAL = 0
    BOLD = auto()
    FAINT = auto()
    ITALIC = auto()
    UNDERLINE = auto()
    BLINK_SLOW = auto()
    BLINK_FAST = auto()
    REVERSE = auto()
    CONCEAL = auto()
    CROSSED_OUT = auto()

    FG_BLACK = 30
    FG_RED = auto()
    FG_GREEN = auto()
    FG_YELLOW = auto()
    FG_BLUE = auto()
    FG_MAGENTA = auto()
    FG_CYAN = auto()
    FG_WHITE = auto()

    BG_BLACK = 40
    BG_RED = auto()
    BG_GREEN = auto()
    BG_YELLOW = auto()
    BG_BLUE = auto()
    BG_MAGENTA = auto()
    BG_CYAN = auto()
    BG_WHITE = auto()

    @staticmethod
    def to_str(sgr: Union[int, str]) -> str:
        if isinstance(sgr, str):
            sgr = SGR[sgr.upper()]
        if isinstance(sgr, SGR):
            return str(sgr.value)
        if isinstance(sgr, int):
            return str(sgr)

    @staticmethod
    def format(text: str, sgr: Iterable[int]) -> str:
        sgr = ';'.join(str(SGR(parameter).value) for parameter in sgr)
        return f"\033[{sgr}m{text}\033[0m"


if __name__ == '__main__':
    for fg_color in range(30, 38):
        for bg_color in range(40, 48):
            text = ' '.join(
                SGR.format(f'{effect}:{fg_color};{bg_color}', (effect, fg_color, bg_color))
                for effect in range(10)
            )
            print(text)