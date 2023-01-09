__all__ = [
    'SGR',
    'get_logger',
]

import getpass
import inspect
import logging
import socket
from enum import IntEnum, auto
from typing import Iterable, TypeVar

# fix logging format
try:
    import lvis  # noqa: F401
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
except ImportError:
    pass

T = TypeVar('T', bound=IntEnum)


class SGR(IntEnum):
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

    @classmethod
    def format(cls: type[T], text: str, sgrs: Iterable[T]) -> str:
        return f"\033[{';'.join(str(sgr.value) for sgr in sgrs)}m{text}\033[m"


# pragma: no cover
class Formatter(logging.Formatter):

    def __init__(self) -> None:
        super().__init__(
            "[%(asctime)s %(process)d:%(thread)d]"
            "[%(filename)s:%(lineno)d %(name)s.%(funcName)s]"
            " %(levelname)s: %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        sgrs: tuple[SGR, ...]
        if record.levelno == logging.DEBUG:
            sgrs = (SGR.FAINT, )
        elif record.levelno == logging.WARNING:
            sgrs = (SGR.BOLD, SGR.FG_YELLOW)
        elif record.levelno == logging.ERROR:
            sgrs = (SGR.BOLD, SGR.FG_RED)
        elif record.levelno == logging.CRITICAL:
            sgrs = (SGR.BOLD, SGR.BLINK_SLOW, SGR.FG_RED)
        else:
            return s
        return SGR.format(s, sgrs)


def get_logger(
    id_: int | None = None,
    file=None,
) -> logging.Logger:
    name = inspect.stack()[1].frame.f_globals.get('__name__')
    if id_ is not None:
        name = f'{name}.{id_}'
    logger = logging.getLogger(name)
    if not getattr(logger, '_initialized', False):
        setattr(logger, '_initialized', True)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())

        if file is not None:
            logger.addHandler(logging.FileHandler(file))

        formatter = Formatter()
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        logger.propagate = False
        logger.debug(
            f"logger initialized by {getpass.getuser()}@{socket.gethostname()}"
        )
    return logger


if __name__ == '__main__':
    for fg_color in range(30, 38):
        for bg_color in range(40, 48):
            text = ' '.join(
                SGR.format(
                    f'{effect}:{fg_color};{bg_color}',
                    map(SGR, (effect, fg_color, bg_color)),
                ) for effect in range(10)
            )
            print(text)
