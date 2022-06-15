import sys
import typing

from typing_extensions import Literal, Protocol

from ..logger import get_logger

if sys.version_info < (3, 8):
    get_logger().warning(
        "Monkey patching `typing.Literal` and `typing.Protocol`.",
    )
    typing.Literal = Literal
    typing.Protocol = Protocol
