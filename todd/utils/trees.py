__all__ = [
    'BaseUtil',
    'TreeUtil',
    'TensorTreeUtil',
]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Set
from functools import partial
from itertools import starmap
from typing import Any, Callable, Generic, Iterable, TypeGuard, TypeVar, cast

import torch

from ..patches.torch import all_close
from ..registries import Registry

T = TypeVar('T')


class CollectionUtilRegistry(Registry):
    pass


class BaseUtil(ABC):

    @classmethod
    @abstractmethod
    def elements(cls, obj: Any) -> list:
        """Get the elements of the given object.

        Args:
            obj: The object to get the elements from.

        Returns:
            Elements of the given object.
        """

    @classmethod
    @abstractmethod
    def map(cls, f: Callable[..., Any], *objs: Any) -> Any:
        """Apply a function to the given objects.

        Args:
            f: The function to apply.
            objs: The objects to apply the function to.

        Returns:
            The result of applying the function to the objects.
        """

    @classmethod
    @abstractmethod
    def reduce(cls, f: Callable[[list], Any], obj: Any) -> Any:
        """Apply a function to the collection and returns a single value.

        Args:
            f: The function to apply to the elements.
            obj: The collection to reduce.

        Returns:
            The result of applying the function to the collection.
        """


class CollectionUtil(BaseUtil, Generic[T], ABC):
    """Utility class for working with collections."""

    @classmethod
    @abstractmethod
    def isinstance(cls, obj: Any) -> TypeGuard[T]:
        """Check if the given object is an instance of the specified type.

        Args:
            obj: The object to check.

        Returns:
            `True` if the object is an instance of the specified type, `False`
            otherwise.
        """

    @classmethod
    @abstractmethod
    def elements(cls, obj: T) -> list:
        pass

    @classmethod
    @abstractmethod
    def map(cls, f: Callable[..., Any], *objs: T) -> T:
        pass

    @classmethod
    def reduce(cls, f: Callable[[list], Any], obj: T) -> Any:
        return f(cls.elements(obj))


@CollectionUtilRegistry.register_()
class MappingUtil(CollectionUtil[Mapping]):
    """Utility class for working with mappings."""

    @classmethod
    def isinstance(cls, obj: Any) -> TypeGuard[Mapping]:
        return isinstance(obj, Mapping)

    @classmethod
    def elements(cls, obj: Mapping) -> list:
        """Get the elements of a `Mapping`.

        Args:
            obj: The object.

        Returns:
            The elements of the `Mapping`.
        """
        return list(obj.values())

    @classmethod
    def map(cls, f: Callable[..., Any], *objs: Mapping) -> dict:
        """Apply a function to the values of multiple mappings.

        Args:
            f: The function to apply.
            objs: The objects.

        Returns:
            A new `dict` with the results of applying the function to the
            values.
        """
        return {k: f(*[o[k] for o in objs]) for k in set().union(*objs)}


@CollectionUtilRegistry.register_()
class SequenceUtil(CollectionUtil[Sequence]):
    """Utility class for working with sequences.

    `str` objects are not supported by this class.
    """

    @classmethod
    def isinstance(cls, obj: Any) -> TypeGuard[Sequence]:
        return isinstance(obj, Sequence) and not isinstance(obj, str)

    @classmethod
    def elements(cls, obj: Sequence) -> list:
        """Elements of the given `Sequence`.

        Args:
            obj: The object.

        Returns:
            The elements of the `Sequence`.
        """
        return list(obj)

    @classmethod
    def map(cls, f: Callable[..., Any], *objs: Sequence) -> tuple:
        """Apply the given function to each element of the sequences.

        Args:
            f: The function to apply.
            objs: The sequences to apply the function to.

        Returns:
            A `tuple` of the results after applying the function to each
            element.
        """
        return tuple(starmap(f, zip(*objs)))  # type: ignore


@CollectionUtilRegistry.register_()
class SetUtil(CollectionUtil[Set]):
    """Utility class for working with sets."""

    @classmethod
    def isinstance(cls, obj: Any) -> TypeGuard[Set]:
        return isinstance(obj, Set) and not isinstance(obj, str)

    @classmethod
    def elements(cls, obj: Set) -> list:
        """Elements of a `Set`.

        Args:
            obj: The `Set` to retrieve elements from.

        Returns:
            The elements of the `Set`.
        """
        return list(obj)

    @classmethod
    def map(cls, f: Callable[..., Any], *objs: Set) -> set:
        """Apply a function to each element of a `Set`.

        Args:
            f: The function to apply.
            objs: The `set` objects to apply the function to.

        Returns:
            A new `set` with the function applied to each element.
        """
        return set(starmap(f, zip(*objs)))  # type: ignore


class TreeUtil(BaseUtil):

    @classmethod
    def get_util(cls, *objs: Any) -> type[CollectionUtil] | None:
        """Find a utility class for all the given object.

        Args:
            objs: The objects to check.

        Returns:
            The utility class or `None` if none of the utility classes is
            applicable.

        Examples:
            >>> TreeUtil.get_util([])
            <class '...SequenceUtil'>
            >>> TreeUtil.get_util(tuple())
            <class '...SequenceUtil'>
            >>> TreeUtil.get_util(dict())
            <class '...MappingUtil'>
            >>> TreeUtil.get_util(set())
            <class '...SetUtil'>
            >>> TreeUtil.get_util('')
            >>> TreeUtil.get_util(123)
            >>> TreeUtil.get_util(None)
        """
        utils = cast(
            Iterable[type[CollectionUtil]],
            CollectionUtilRegistry.values(),
        )
        utils = set(u for u in utils if all(map(u.isinstance, objs)))
        if len(utils) == 0:
            return None
        util, = utils
        return util

    @classmethod
    def is_atom(cls, obj: Any) -> bool:
        return cls.get_util(obj) is None

    @classmethod
    def is_atomic(cls, obj: Any) -> bool:
        """Check if the given object is atomic.

        An object is considered atomic if all its elements are not collections.

        Args:
            obj: The object to check.

        Returns:
            `True` if the object is atomic, `False` otherwise.

        Examples:
            >>> TreeUtil.is_atomic([])
            True
            >>> TreeUtil.is_atomic({1: 'a', 2: 'b'})
            True
            >>> TreeUtil.is_atomic({1, 2, 3})
            True
            >>> TreeUtil.is_atomic(('a', 'b', 'c'))
            True
            >>> TreeUtil.is_atomic([1, [2, 3], [4, [5, 6]]])
            False
            >>> TreeUtil.is_atomic({1: [2, 3], 4: [5, 6]})
            False
        """
        util = cls.get_util(obj)
        return util is not None and all(map(cls.is_atom, util.elements(obj)))

    @classmethod
    def map(cls, f: Callable[..., Any], *objs: Any) -> Any:
        """Recursively apply a function to the given objects.

        Args:
            f: The function to apply.
            objs: The objects to apply the function to.

        Returns:
            The result of applying the function to the objects.

        Examples:
            >>> TreeUtil.map(lambda x: x + 1, [1, 2, 3])
            (2, 3, 4)
            >>> result = TreeUtil.map(
            ...     lambda x: x.upper(),
            ...     {'a': 'apple', 'b': 'banana'},
            ... )
            >>> dict(sorted(result.items()))
            {'a': 'APPLE', 'b': 'BANANA'}
            >>> TreeUtil.map(lambda x: x * 2, set([1, 2, 3]))
            {2, 4, 6}
            >>> TreeUtil.map(lambda x: x.lower(), ('HELLO', 'WORLD'))
            ('hello', 'world')
        """
        util = cls.get_util(*objs)
        return (
            f(*objs) if util is None else util.map(partial(cls.map, f), *objs)
        )

    @classmethod
    def reduce(cls, f: Callable[[list], Any], obj: Any) -> Any:
        """Apply a function to the collection and returns a single value.

        Args:
            f: The function to apply to the elements.
            obj: The collection to reduce.

        Returns:
            The result of applying the function to the collection.

        Examples:
            >>> TreeUtil.reduce(sum, [])
            0
            >>> TreeUtil.reduce(sum, [1])
            1
            >>> TreeUtil.reduce(sum, [1, 2, 3, 4])
            10
            >>> TreeUtil.reduce(sum, [[1, 2], [3, 4], [5, 6]])
            21
        """
        util = cls.get_util(obj)
        if util is None:
            return obj
        obj = util.map(partial(cls.reduce, f), obj)
        obj = util.reduce(f, obj)
        return obj

    @classmethod
    def elements(cls, obj: Any) -> list:
        """Get the elements of the given object.

        Args:
            obj: The object to get the elements from.

        Returns:
            Elements of the given object.

        Examples:
            >>> list(TreeUtil.elements([]))
            []
            >>> list(TreeUtil.elements({1: 'a', 2: 'b'}))
            ['a', 'b']
            >>> list(TreeUtil.elements({1, 2, 3}))
            [1, 2, 3]
            >>> list(TreeUtil.elements(('a', 'b', 'c')))
            ['a', 'b', 'c']
            >>> TreeUtil.elements([[1, 2], [3, [4, 5]]])
            [1, 2, 3, 4, 5]
            >>> TreeUtil.elements([1, {2: 'a'}, (3, 4)])
            [1, 'a', 3, 4]
        """
        util = cls.get_util(obj)
        if util is None:
            return [obj]
        obj = util.elements(obj)
        assert SequenceUtil.isinstance(obj)
        obj = SequenceUtil.map(cls.elements, obj)
        obj = SequenceUtil.reduce(partial(sum, start=[]), obj)
        return obj

    @classmethod
    def index(cls, obj: Any, index: Any) -> Any:
        for i in index:
            obj = obj[i]
        return obj


class TensorTreeUtil(TreeUtil):

    @classmethod
    def all_close(cls, x: Any, y: Any, **kwargs) -> bool:
        return cls.reduce(all, cls.map(partial(all_close, **kwargs), x, y))

    @classmethod
    def stack(cls, obj: Any, **kwargs) -> torch.Tensor:
        return cls.reduce(partial(torch.stack, **kwargs), obj)

    @classmethod
    def new_empty(cls, obj: Any, *args, **kwargs) -> torch.Tensor:
        util = cls.get_util(obj)
        if util is None:
            assert isinstance(obj, torch.Tensor)
            return obj.new_empty(*args, **kwargs)
        elements = util.elements(obj)
        return cls.new_empty(elements[0], *args, **kwargs)

    @classmethod
    def shape(cls, obj: Any, depth: int = 0) -> tuple[int, ...]:
        util = cls.get_util(obj)
        if util is None:
            assert isinstance(obj, torch.Tensor)
            return obj.shape[max(depth, 0):]
        elements = util.elements(obj)
        shape, = {cls.shape(f, depth - 1) for f in elements}
        if depth <= 0:
            shape = (len(elements), ) + shape
        return shape

    @classmethod
    def index(cls, obj: Any, index: torch.Tensor) -> torch.Tensor:
        m, n = index.shape
        if m == 0:
            shape = cls.shape(obj, n)
            return cls.new_empty(obj, m, *shape)
        if n == 0:
            tensor = cls.stack(obj)
            return tensor.unsqueeze(0).repeat(m, *[1] * tensor.ndim)
        super_index = super().index
        return cls.stack([super_index(obj, i) for i in index.int().tolist()])
