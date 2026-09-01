__all__ = [
    'CSI',
    'control_sequence',
]

from typing import Any, Iterable

CSI = '\033['


def control_sequence(
    parameter_bytes: Iterable[Any],
    intermediate_bytes: Iterable[Any],
    final_byte: str,
) -> str:
    return (
        CSI + ';'.join(map(str, parameter_bytes))
        + ''.join(map(str, intermediate_bytes)) + final_byte
    )
