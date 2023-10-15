__all__ = [
    'Formatter',
    'SGRFormatter',
    'logger',
]

import logging

from ..utils import SGR


class Formatter(logging.Formatter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            "[%(asctime)s %(process)d:%(thread)d]"
            "[%(filename)s:%(lineno)d %(name)s %(funcName)s]"
            " %(levelname)s: %(message)s",
            *args,
            **kwargs,
        )


class SGRFormatter(Formatter):

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if record.levelno == logging.DEBUG:
            return SGR.format(s, SGR.FAINT)
        if record.levelno == logging.WARNING:
            return SGR.format(s, SGR.BOLD, SGR.FG_YELLOW)
        if record.levelno == logging.ERROR:
            return SGR.format(s, SGR.BOLD, SGR.FG_RED)
        if record.levelno == logging.CRITICAL:
            return SGR.format(s, SGR.BOLD, SGR.BLINK_SLOW, SGR.FG_RED)
        return s


try:
    # prevent lvis from overriding the logging config
    import lvis  # noqa: F401 pylint: disable=unused-import
except ImportError:
    pass
logging.basicConfig(force=True)

logger = logging.getLogger('todd')
logger.propagate = False
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(SGRFormatter())
logger.addHandler(handler)
