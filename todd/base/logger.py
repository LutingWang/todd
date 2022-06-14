import getpass
import logging
import os
import socket
from abc import abstractmethod
from enum import IntEnum, auto
from typing import Iterable, Optional

__all__ = [
    'SGR',
    'get_logger',
]

_logger_initialized = False


class ANSI(IntEnum):  # fix bug in python<=3.7.1

    @classmethod
    def to_str(cls, key) -> str:
        if isinstance(key, str):
            return str(cls[key.upper()].value)
        if isinstance(key, cls):
            return str(key.value)
        return str(int(key))

    @staticmethod
    @abstractmethod
    def format(text: str) -> str:
        pass


class SGR(ANSI):
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
    def format(
        cls,
        text: str,
        sgr: Iterable = tuple(),
    ) -> str:
        sgr_list = ';'.join(map(cls.to_str, sgr))
        return f"\033[{sgr_list}m{text}\033[m"


def get_logger(log_file: Optional[str] = None, level: str = 'DEBUG'):
    global _logger_initialized
    if not _logger_initialized:

        _logger_initialized = True
        logger = logging.getLogger('Todd')
        logger.setLevel(getattr(logging, level))
        worker_pid = (
            f"{getpass.getuser()}@{socket.gethostname()}:{os.getpid()}"
        )
        formatter = logging.Formatter(
            fmt=(  # yapf: disable
                f"[{worker_pid:s}][%(asctime)s]"
                "[%(filename)s:%(funcName)s:%(lineno)d] "
                + SGR.format(
                    "%(name)s %(levelname)s:", sgr=(SGR.BOLD, SGR.FG_BLUE),
                ) +
                "\n%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False
    return logging.getLogger('Todd')


if __name__ == '__main__':
    for fg_color in range(30, 38):
        for bg_color in range(40, 48):
            text = ' '.join(
                SGR.format(
                    f'{effect}:{fg_color};{bg_color}',
                    (effect, fg_color, bg_color),
                ) for effect in range(10)
            )
            print(text)
