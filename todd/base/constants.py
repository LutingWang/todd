__all__ = [
    'SGR',
    'Color',
    'Colors',
    'FileType',
]

import enum
from typing import Iterable, NamedTuple, TypeVar

SGRType = TypeVar('SGRType', bound='SGR')


class SGR(enum.IntEnum):
    """Select Graphic Rendition.

    Refer to https://en.wikipedia.org/wiki/ANSI_escape_code.
    """
    NORMAL = 0
    BOLD = enum.auto()
    FAINT = enum.auto()
    ITALIC = enum.auto()
    UNDERLINE = enum.auto()
    BLINK_SLOW = enum.auto()
    BLINK_FAST = enum.auto()
    REVERSE = enum.auto()
    CONCEAL = enum.auto()
    CROSSED_OUT = enum.auto()

    FG_BLACK = 30
    FG_RED = enum.auto()
    FG_GREEN = enum.auto()
    FG_YELLOW = enum.auto()
    FG_BLUE = enum.auto()
    FG_MAGENTA = enum.auto()
    FG_CYAN = enum.auto()
    FG_WHITE = enum.auto()

    BG_BLACK = 40
    BG_RED = enum.auto()
    BG_GREEN = enum.auto()
    BG_YELLOW = enum.auto()
    BG_BLUE = enum.auto()
    BG_MAGENTA = enum.auto()
    BG_CYAN = enum.auto()
    BG_WHITE = enum.auto()

    @classmethod
    def CSI(cls: type[SGRType], parameters: Iterable[SGRType]) -> str:
        """Control Sequence Introducer."""
        return f'\033[{";".join(str(p.value) for p in parameters)}m'

    @classmethod
    def format(cls: type[SGRType], text: str, *args: SGRType) -> str:
        return cls.CSI(args) + text + cls.CSI(tuple())


class Color(NamedTuple):
    red: float
    green: float
    blue: float


class Colors:
    palette = [
        Color(106, 0, 228),
        Color(119, 11, 32),
        Color(165, 42, 42),
        Color(0, 0, 192),
        Color(197, 226, 255),
        Color(0, 60, 100),
        Color(0, 0, 142),
        Color(255, 77, 255),
        Color(153, 69, 1),
        Color(120, 166, 157),
        Color(0, 182, 199),
        Color(0, 226, 252),
        Color(182, 182, 255),
        Color(0, 0, 230),
        Color(220, 20, 60),
        Color(163, 255, 0),
        Color(0, 82, 0),
        Color(3, 95, 161),
        Color(0, 80, 100),
        Color(183, 130, 88),
    ]

    def __class_getitem__(cls, index: int) -> Color:
        index %= len(cls.palette)
        return cls.palette[index]


class FileType(str, enum.Enum):
    TEXT = '.txt'
    HTML = '.html'
    PYTHON = '.py'
    TOML = '.toml'


if __name__ == '__main__':
    import itertools

    for fg, bg in itertools.product(range(30, 38), range(40, 48)):
        text = ' '.join(
            SGR.format(f'{effect}:{fg};{bg}', SGR(effect), SGR(fg), SGR(bg))
            for effect in range(10)
        )
        print(text)
