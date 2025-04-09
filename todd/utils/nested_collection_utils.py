__all__ = [
    'CallableProtocol',
    'HandlerRegistry',
    'BaseHandler',
    'MappingHandler',
    'SequenceHandler',
    'SetHandler',
    'NestedCollectionUtils',
    'NestedTensorCollectionUtils',
]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Set
from functools import partial
from itertools import starmap
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
)

import einops
import torch

from ..bases.registries import Registry
from ..patches.torch import all_close

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)
T_co = TypeVar('T_co', covariant=True)
KT = TypeVar('KT')
VT = TypeVar('VT')


class CallableProtocol(Protocol[T_contra, T_co]):

    def __call__(self, *args: T_contra) -> T_co:
        ...


class HandlerRegistry(Registry):
    pass


class BaseHandler(Generic[T], ABC):
    """Utility class for working with collections."""

    @classmethod
    @abstractmethod
    def can_handle(cls, obj: Any) -> TypeGuard[T]:
        pass

    @classmethod
    @abstractmethod
    def elements(cls, obj: T) -> list[Any]:
        pass

    @classmethod
    @abstractmethod
    def map(cls, f: CallableProtocol[Any, Any], *objs: T) -> T:
        """Apply a function to the given objects.

        All the objects must be of the same shape.
        """


@HandlerRegistry.register_()
class MappingHandler(BaseHandler[Mapping[KT, VT]]):

    @classmethod
    def can_handle(cls, obj: Any) -> TypeGuard[Mapping[KT, VT]]:
        return isinstance(obj, Mapping)

    @classmethod
    def elements(cls, obj: Mapping[KT, VT]) -> list[VT]:
        return list(obj.values())

    @classmethod
    def map(
        cls,
        f: CallableProtocol[VT, T_co],
        *objs: Mapping[KT, VT],
    ) -> dict[KT, T_co]:
        return {k: f(*[o[k] for o in objs]) for k in set().union(*objs)}


@HandlerRegistry.register_()
class SequenceHandler(BaseHandler[Sequence[T]]):

    @classmethod
    def can_handle(cls, obj: Any) -> TypeGuard[Sequence[T]]:
        return isinstance(obj, Sequence)

    @classmethod
    def elements(cls, obj: Sequence[T]) -> list[T]:
        return list(obj)

    @classmethod
    def map(
        cls,
        f: CallableProtocol[T, T_co],
        *objs: Sequence[T],
    ) -> tuple[T_co, ...]:
        return tuple(starmap(f, zip(*objs)))


@HandlerRegistry.register_()
class SetHandler(BaseHandler[Set[T]]):

    @classmethod
    def can_handle(cls, obj: Any) -> TypeGuard[Set[T]]:
        return isinstance(obj, Set)

    @classmethod
    def elements(cls, obj: Set[T]) -> list[T]:
        return list(obj)

    @classmethod
    def map(cls, f: CallableProtocol[T, T_co], *objs: Set[T]) -> set[T_co]:
        return set(starmap(f, zip(*objs)))


class NestedCollectionUtils:

    def __init__(
        self,
        *args,
        atomic_types: Iterable[type[Any]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if atomic_types is None:
            atomic_types = [str]
        self._atomic_types = tuple(atomic_types)

    def add_atomic_type(self, *types: type[Any]) -> None:
        self._atomic_types = tuple(set(self._atomic_types) | set(types))

    def get_handler(self, *objs: Any) -> type[BaseHandler[Any]] | None:
        """Find a utility class for all the given object.

        Args:
            objs: The objects to check.

        Returns:
            The utility class or `None` if none of the utility classes is
            applicable.

        Examples:
            >>> utils = NestedCollectionUtils()
            >>> utils.get_handler([])
            <class '...SequenceHandler'>
            >>> utils.get_handler(tuple())
            <class '...SequenceHandler'>
            >>> utils.get_handler(dict())
            <class '...MappingHandler'>
            >>> utils.get_handler(set())
            <class '...SetHandler'>
            >>> utils.get_handler('')
            >>> utils.get_handler(123)
            >>> utils.get_handler(None)
        """
        if any(isinstance(obj, self._atomic_types) for obj in objs):
            return None
        handlers = set(
            cast(
                Iterable[type[BaseHandler[Any]]],
                HandlerRegistry.values(),
            ),
        )
        handlers = set(
            handler for handler in handlers
            if all(map(handler.can_handle, objs))
        )
        if not handlers:
            return None
        handler, = handlers
        return handler

    def can_handle(self, obj: Any) -> bool:
        return self.get_handler(obj) is not None

    def is_atomic(self, obj: Any) -> bool:
        return self.get_handler(obj) is None

    def is_atomic_collection(self, obj: Any) -> bool:
        """Check if the given object is atomic.

        An object is considered atomic if all its elements are not collections.

        Args:
            obj: The object to check.

        Returns:
            `True` if the object is atomic, `False` otherwise.

        Examples:
            >>> utils = NestedCollectionUtils()
            >>> utils.is_atomic_collection([])
            True
            >>> utils.is_atomic_collection({1: 'a', 2: 'b'})
            True
            >>> utils.is_atomic_collection({1, 2, 3})
            True
            >>> utils.is_atomic_collection(('a', 'b', 'c'))
            True
            >>> utils.is_atomic_collection([1, [2, 3], [4, [5, 6]]])
            False
            >>> utils.is_atomic_collection({1: [2, 3], 4: [5, 6]})
            False
        """
        handler = self.get_handler(obj)
        return (
            handler is not None
            and all(map(self.is_atomic, handler.elements(obj)))
        )

    def elements(self, obj: Any) -> list[Any]:
        """Get the elements of the given object.

        Args:
            obj: The object to get the elements from.

        Returns:
            Elements of the given object.

        Examples:
            >>> utils = NestedCollectionUtils()
            >>> list(utils.elements([]))
            []
            >>> list(utils.elements({1: 'a', 2: 'b'}))
            ['a', 'b']
            >>> list(utils.elements({1, 2, 3}))
            [1, 2, 3]
            >>> list(utils.elements(('a', 'b', 'c')))
            ['a', 'b', 'c']
            >>> utils.elements([[1, 2], [3, [4, 5]]])
            [1, 2, 3, 4, 5]
            >>> utils.elements([1, {2: 'a'}, (3, 4)])
            [1, 'a', 3, 4]
        """
        handler = self.get_handler(obj)
        if handler is None:
            return [obj]
        elements = handler.elements(obj)
        elements = [self.elements(e) for e in elements]
        return sum(elements, [])

    def map(self, f: CallableProtocol[Any, Any], *objs: Any) -> Any:
        """Recursively apply a function to the given objects.

        Args:
            f: The function to apply.
            objs: The objects to apply the function to.

        Returns:
            The result of applying the function to the objects.

        Examples:
            >>> utils = NestedCollectionUtils()
            >>> utils.map(lambda x: x + 1, [1, 2, 3])
            (2, 3, 4)
            >>> result = utils.map(
            ...     lambda x: x.upper(),
            ...     {'a': 'apple', 'b': 'banana'},
            ... )
            >>> dict(sorted(result.items()))
            {'a': 'APPLE', 'b': 'BANANA'}
            >>> utils.map(lambda x: x * 2, set([1, 2, 3]))
            {2, 4, 6}
            >>> utils.map(lambda x: x.lower(), ('HELLO', 'WORLD'))
            ('hello', 'world')
        """
        handler = self.get_handler(*objs)
        if handler is None:
            return f(*objs)
        f = partial(self.map, f)
        return handler.map(f, *objs)

    def reduce(self, f: Callable[[Iterable[Any]], Any], obj: Any) -> Any:
        """Apply a function to the collection and returns a single value.

        Args:
            f: The function to apply to the elements.
            obj: The collection to reduce.

        Returns:
            The result of applying the function to the collection.

        Examples:
            >>> utils = NestedCollectionUtils()
            >>> utils.reduce(sum, [])
            0
            >>> utils.reduce(sum, [1])
            1
            >>> utils.reduce(sum, [1, 2, 3, 4])
            10
            >>> utils.reduce(sum, [[1, 2], [3, 4], [5, 6]])
            21
        """
        handler = self.get_handler(obj)
        if handler is None:
            return obj
        elements = handler.elements(obj)
        elements = [self.reduce(f, e) for e in elements]
        return f(elements)

    def index(self, obj: Any, indices: Any) -> Any:
        """Index the given object with the given indices.

        Args:
            obj: The object to index.
            indices: The indices to use for indexing.

        Returns:
            The result of indexing the object with the indices.

        Examples:
            >>> utils = NestedCollectionUtils()
            >>> utils.index([1, 2, 3], 0)
            1
            >>> utils.index({1: 'a', 2: 'b', 3: 'c'}, 1)
            'a'
            >>> utils.index([[1, 2], [3, 4], [5, 6]], [0, 1])
            2
        """
        if not isinstance(indices, Iterable):
            return obj[indices]
        for index in indices:
            obj = obj[index]
        return obj


class NestedTensorCollectionUtils(NestedCollectionUtils):

    def all_close(self, x: Any, y: Any, **kwargs) -> bool:
        f = partial(all_close, **kwargs)
        return self.reduce(all, self.map(f, x, y))

    def stack(self, obj: Any, **kwargs) -> torch.Tensor:
        f = partial(torch.stack, **kwargs)
        return self.reduce(f, obj)

    def new_empty(self, obj: Any, *args, **kwargs) -> torch.Tensor:
        handler = self.get_handler(obj)
        if handler is None:
            assert isinstance(obj, torch.Tensor)
            return obj.new_empty(*args, **kwargs)
        elements = handler.elements(obj)
        return self.new_empty(elements[0], *args, **kwargs)

    # TODO: support range depth
    def shape(self, obj: Any, depth: int = 0) -> tuple[int, ...]:
        handler = self.get_handler(obj)
        if handler is None:
            assert isinstance(obj, torch.Tensor)
            return obj.shape[max(depth, 0):]
        elements = handler.elements(obj)
        shape, = {self.shape(f, depth - 1) for f in elements}
        if depth <= 0:
            shape = (len(elements), ) + shape
        return shape

    def index(self, obj: Any, indices: torch.Tensor) -> torch.Tensor:
        m, n = indices.shape
        if m == 0:
            shape = self.shape(obj, n)
            return self.new_empty(obj, m, *shape)
        if n == 0:
            tensor = self.stack(obj)
            return einops.repeat(tensor, '... -> m ...', m=m)
        super_index = super().index
        return self.stack([
            super_index(obj, index) for index in indices.int().tolist()
        ])
