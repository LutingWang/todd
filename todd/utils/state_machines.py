__all__ = [
    'StateMachine',
]

import enum
from typing import Generic, Mapping, TypeVar

T = TypeVar('T', bound=enum.Enum)


class StateMachine(Generic[T]):
    """A class to represent a state that can be transited to another state.

    Examples:
        >>> class Enum(enum.Enum):
        ...     A = 1
        ...     B = 2
        ...     C = 3
        >>> state = StateMachine(Enum.A)
        >>> state.state
        <Enum.A: 1>
        >>> state.transit({Enum.A: Enum.B})
        >>> state.state
        <Enum.B: 2>
        >>> state.transit({Enum.A: Enum.C})
        Traceback (most recent call last):
            ...
        RuntimeError: Enum.B is not in {<Enum.A: 1>: <Enum.C: 3>}.
    """

    def __init__(self, state: T) -> None:
        self._state = state

    @property
    def state(self) -> T:
        return self._state

    def transit(self, transitions: Mapping[T, T]) -> None:
        if self._state not in transitions:
            raise RuntimeError(f"{self._state} is not in {transitions}.")
        self._state = transitions[self._state]
