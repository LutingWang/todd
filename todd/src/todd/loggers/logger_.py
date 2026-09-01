__all__ = [
    'logger',
]

import logging

from .formatter import Formatter

logger = logging.getLogger('todd')
logger.propagate = False
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(Formatter())
logger.addHandler(handler)
