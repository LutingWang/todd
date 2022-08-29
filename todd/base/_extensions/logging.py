__all__ = [
    'SGR',
    'log_file_initialized',
    'init_log_file',
    'get_log_file',
    'get_logger',
]

import getpass
import inspect
import logging
import os
import socket
from abc import abstractmethod
from enum import IntEnum, auto
from typing import Iterable


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


DEFAULT_FORMAT = (
    "[%(asctime)s]"
    "[Todd %(name)s]"
    "[%(filename)s:%(funcName)s:%(lineno)d]"
    " %(levelname)s: %(message)s"
)


# pragma: no cover
class Formatter(logging.Formatter):
    FORMATTERS = dict(
        zip(
            (
                logging.DEBUG,
                logging.WARNING,
                logging.ERROR,
                logging.CRITICAL,
            ),
            map(
                lambda sgr: logging.Formatter(
                    SGR.format(DEFAULT_FORMAT, sgr),
                ),
                (
                    (SGR.FAINT, ),
                    (SGR.BOLD, SGR.FG_YELLOW),
                    (SGR.BOLD, SGR.FG_RED),
                    (SGR.BOLD, SGR.BLINK_SLOW, SGR.FG_RED),
                ),
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__(DEFAULT_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        formatter = self.FORMATTERS.get(record.levelno)
        if formatter is None:
            return super().format(record)
        return formatter.format(record)


_log_file = None


def log_file_initialized() -> bool:
    global _log_file
    return _log_file is not None


def init_log_file(log_file) -> None:
    global _log_file
    if _log_file is not None and log_file is not None:
        get_logger().warning(
            f"log_file={_log_file} has been reset to {log_file}.",
        )
    _log_file = log_file


def get_log_file() -> str:
    global _log_file
    assert _log_file is not None
    return _log_file


def get_logger() -> logging.Logger:
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    assert frame is not None
    name = frame.f_globals.get('__name__')
    logger = logging.getLogger(name)
    if not getattr(logger, '_isinitialized', False):
        setattr(logger, '_isinitialized', True)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        if log_file_initialized():
            logger.addHandler(logging.FileHandler(get_log_file()))
        formatter = Formatter()
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        logger.propagate = False
        logger.debug(
            f"logger initialized by"
            f" {getpass.getuser()}"
            f"@{socket.gethostname()}"
            f":{os.getpid()}"
        )
    return logger


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
