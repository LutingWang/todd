__all__ = [
    'MappingUtil',
    'SetUtil',
    'SequenceUtil',
    'CollectionUtil',
    'TensorCollectionUtil',
]

from typing import Callable, Collection, Mapping, Sequence, Set, TypeVar

import torch

from ..utils import NestedStrDict, NestedStrMapping, StrDict, StrMapping

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')


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
                sd |= MappingUtil.flatten(v, k)
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


class CollectionUtil:

    @classmethod
    def map(
        cls,
        collection: Collection[VT] | VT,
        function: Callable[[VT], T],
    ) -> Collection[T] | T:
        if isinstance(collection, Mapping):
            return MappingUtil.map(collection, function)
        if isinstance(collection, Set):
            return SetUtil.map(collection, function)
        if isinstance(collection, Sequence):
            return SequenceUtil.map(collection, function)
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
