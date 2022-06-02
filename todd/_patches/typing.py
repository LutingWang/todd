from typing_extensions import Literal
import sys
import typing

from todd.logger import get_logger
_logger = get_logger()


if sys.version_info < (3, 8):
    _logger.warning("Monkey patching `typing.Literal`.")
    typing.Literal = Literal
