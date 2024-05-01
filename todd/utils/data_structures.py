__all__ = [
    'PriorityQueue',
    'CollectionUtil',
    'TensorCollectionUtil',
    'Status',
]

import enum
from collections import UserList
from typing import (
    Callable,
    Collection,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    Set,
    TypeVar,
)

import torch

from .type_aliases import NestedStrDict, NestedStrMapping, StrDict, StrMapping

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
EnumType = TypeVar('EnumType', bound=enum.Enum)  # pylint: disable=invalid-name


class PriorityQueue(UserList[tuple[Mapping[KT, int], VT]]):

    def __init__(
        self,
        priorities: Iterable[Mapping[KT, int]],
        queue: Iterable[VT],
    ) -> None:
        priorities = list(priorities)
        queue = list(queue)
        super().__init__(zip(priorities, queue))

    @property
    def priorities(self) -> list[Mapping[KT, int]]:
        return [p for p, _ in self]

    @property
    def queue(self) -> list[VT]:
        return [q for _, q in self]

    def __call__(self, key: KT) -> list[VT]:
        if len(self) == 0:
            return []
        priorities = [p.get(key, 0) for p in self.priorities]
        priority_index = [(p, i) for i, p in enumerate(priorities)]
        priority_index = sorted(priority_index)
        _, indices = zip(*priority_index)
        queue = [self.queue[i] for i in indices]
        return queue


class CollectionUtil:

    class MappingUtil:

        @classmethod
        def flatten(
            cls,
            nsm: NestedStrMapping[VT],
            prefix: str | None = None,
            delimiter: str = '/',
        ) -> StrDict[VT]:
            sd: StrDict[VT] = dict()
            for k, v in nsm.items():
                if prefix is not None:
                    k = prefix + delimiter + k
                if isinstance(v, Mapping):
                    sd |= CollectionUtil.MappingUtil.flatten(v, k)
                else:
                    sd[k] = v
            return sd

        @classmethod
        def unflatten(
            cls,
            sm: StrMapping[VT],
            delimiter: str = '/',
        ) -> NestedStrDict[VT]:
            nsd: NestedStrDict[VT] = dict()
            for k, v in sm.items():
                *prefixes, k = k.split(delimiter)
                sd = nsd
                for prefix in prefixes:
                    sd.setdefault(prefix, dict())
                    sd = sd[prefix]  # type: ignore[assignment]
                sd[k] = v
            return nsd

        @classmethod
        def map(
            cls,
            mapping: Mapping[KT, VT],
            function: Callable[[VT], T],
        ) -> dict[KT, T]:
            return {k: function(v) for k, v in mapping.items()}

    class SetUtil:

        @classmethod
        def map(cls, set_: Set[VT], function: Callable[[VT], T]) -> set[T]:
            return set(map(function, set_))

    class SequenceUtil:

        @classmethod
        def map(
            cls,
            sequence: Sequence[VT],
            function: Callable[[VT], T],
        ) -> list[T]:
            return list(map(function, sequence))

    @classmethod
    def map(
        cls,
        collection: Collection[VT] | VT,
        function: Callable[[VT], T],
    ) -> Collection[T] | T:
        if isinstance(collection, Mapping):
            return cls.MappingUtil.map(collection, function)
        if isinstance(collection, Set):
            return cls.SetUtil.map(collection, function)
        if isinstance(collection, Sequence):
            return cls.SequenceUtil.map(collection, function)
        assert not isinstance(collection, Collection)
        return function(collection)


class TensorCollectionUtil(CollectionUtil):
    TensorCollection = Collection[torch.Tensor]

    def __init__(self, method: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hasattr(torch.Tensor, method)
        self._method = method

    def __call__(
        self,
        tensor_collection: TensorCollection | torch.Tensor,
        *args,
        **kwargs,
    ) -> TensorCollection | torch.Tensor:

        def f(t: torch.Tensor) -> torch.Tensor:
            return getattr(t, self._method)(*args, **kwargs)

        return self.map(tensor_collection, f)


class Status(Generic[EnumType]):

    def __init__(self, status: EnumType) -> None:
        self._status = status

    @property
    def status(self) -> EnumType:
        return self._status

    def transit(self, transitions: Mapping[EnumType, EnumType]) -> None:
        if self._status not in transitions:
            raise RuntimeError(f"{self._status} is not in {transitions}.")
        self._status = transitions[self._status]
