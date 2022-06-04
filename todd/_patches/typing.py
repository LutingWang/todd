import sys
import typing

from typing_extensions import Literal

from .logging import get_logger

_logger = get_logger()

if sys.version_info < (3, 8):
    _logger.warning("Monkey patching `typing.Literal`.")
    typing.Literal = Literal
