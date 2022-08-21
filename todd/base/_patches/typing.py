import sys
import typing

from typing_extensions import Literal, Protocol, TypeGuard, runtime_checkable

from .._extensions import get_logger

if sys.version_info < (3, 8):
    get_logger().warning(
        "Monkey patching `typing.Literal`, `typing.Protocol`, "
        "and `typing.runtime_checkable`.",
    )
    typing.Literal = Literal
    typing.Protocol = Protocol
    typing.runtime_checkable = runtime_checkable

if sys.version_info < (3, 10):
    get_logger().warning("Monkey patching `typing.TypeGuard`.")
    typing.TypeGuard = TypeGuard
