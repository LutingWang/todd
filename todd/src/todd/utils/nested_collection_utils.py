__all__ = [
    'CollectionRegistry',
    'Collection',
    'collection_map',
    'collection_flatten',
    'collection_reduce',
    'collection_index',
]

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, cast

from ..registries import Registry


class CollectionRegistry(Registry):
    pass


class Collection(ABC):

    @classmethod
    @abstractmethod
    def _flatten(cls, obj: Any) -> Iterable[Any]:
        pass

    @classmethod
    @abstractmethod
    def map(cls, f: Callable[[Any], Any], obj: Any) -> Any:
        pass

    @classmethod
    def flatten(cls, obj: Any) -> list[Any]:
        return [
            leaf for child in cls._flatten(obj)
            for leaf in collection_flatten(child)
        ]

    @classmethod
    def reduce(
        cls,
        f: Callable[[Iterable[Any]], Any],
        obj: Any,
    ) -> Any:
        return f(collection_reduce(f, child) for child in cls._flatten(obj))


class DefaultCollection(Collection):

    @classmethod
    def _flatten(cls, obj: Any) -> Iterable[Any]:
        return tuple()

    @classmethod
    def map(cls, f: Callable[[Any], Any], obj: Any) -> Any:
        return f(obj)

    @classmethod
    def flatten(cls, obj: Any) -> list[Any]:
        return [obj]

    @classmethod
    def reduce(
        cls,
        f: Callable[[Iterable[Any]], Any],
        obj: Any,
    ) -> Any:
        return obj


@CollectionRegistry.register_(dict.__name__)
class DictCollection(Collection):

    @classmethod
    def _flatten(cls, obj: dict[Any, Any]) -> Iterable[Any]:
        return obj.values()

    @classmethod
    def map(
        cls,
        f: Callable[[Any], Any],
        obj: dict[Any, Any],
    ) -> dict[Any, Any]:
        return dict(
            zip(
                obj,
                (collection_map(f, child) for child in cls._flatten(obj)),
                strict=True,
            ),
        )


@CollectionRegistry.register_(list.__name__)
class ListCollection(Collection):

    @classmethod
    def _flatten(cls, obj: list[Any]) -> Iterable[Any]:
        return obj

    @classmethod
    def map(cls, f: Callable[[Any], Any], obj: list[Any]) -> list[Any]:
        return [collection_map(f, child) for child in cls._flatten(obj)]


@CollectionRegistry.register_(tuple.__name__)
class TupleCollection(Collection):

    @classmethod
    def _flatten(cls, obj: tuple[Any, ...]) -> Iterable[Any]:
        return obj

    @classmethod
    def map(
        cls,
        f: Callable[[Any], Any],
        obj: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        return tuple(collection_map(f, child) for child in cls._flatten(obj))


@CollectionRegistry.register_(set.__name__)
class SetCollection(Collection):

    @classmethod
    def _flatten(cls, obj: set[Any]) -> Iterable[Any]:
        return obj

    @classmethod
    def map(cls, f: Callable[[Any], Any], obj: set[Any]) -> set[Any]:
        return {collection_map(f, child) for child in cls._flatten(obj)}


def collection_map(f: Callable[[Any], Any], obj: Any) -> Any:
    collection = cast(
        type[Collection],
        CollectionRegistry.get(obj.__class__.__name__, DefaultCollection),
    )
    return collection.map(f, obj)


def collection_flatten(obj: Any) -> list[Any]:
    collection = cast(
        type[Collection],
        CollectionRegistry.get(obj.__class__.__name__, DefaultCollection),
    )
    return collection.flatten(obj)


def collection_reduce(
    f: Callable[[Iterable[Any]], Any],
    obj: Any,
) -> Any:
    collection = cast(
        type[Collection],
        CollectionRegistry.get(obj.__class__.__name__, DefaultCollection),
    )
    return collection.reduce(f, obj)


def collection_index(obj: Any, indices: Any) -> Any:
    if not isinstance(indices, Iterable) or isinstance(indices, (str, bytes)):
        return obj[indices]
    for index in indices:
        obj = obj[index]
    return obj
