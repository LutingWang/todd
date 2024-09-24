__all__ = [
    'Formatter',
]

import logging

from ..patches.py_ import Formatter as BaseFormatter
from .control_sequences import SGR, apply_sgr


class Formatter(BaseFormatter):

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if record.levelno == logging.DEBUG:
            return apply_sgr(s, SGR.FAINT)
        if record.levelno == logging.WARNING:
            return apply_sgr(s, SGR.BOLD, SGR.DISPLAY_YELLOW)
        if record.levelno == logging.ERROR:
            return apply_sgr(s, SGR.BOLD, SGR.DISPLAY_RED)
        if record.levelno == logging.CRITICAL:
            return apply_sgr(s, SGR.BOLD, SGR.SLOWLY_BLINKING, SGR.DISPLAY_RED)
        return s
