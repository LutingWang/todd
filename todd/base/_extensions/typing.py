from typing import Iterable, Iterator, Optional, Protocol, TypeVar

__all__ = [
    'SequenceProto',
]

T_co = TypeVar('T_co', covariant=True)


class SequenceProto(Protocol[T_co]):

    def __init__(self, _: Optional[Iterable[T_co]] = None) -> None:
        ...

    def __getitem__(self, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T_co]:
        ...
