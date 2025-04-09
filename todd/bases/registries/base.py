# pylint: disable=no-value-for-parameter

__all__ = [
    'Item',
    'RegistryMeta',
    'Registry',
]

from collections import UserDict
from typing import Any, Callable, Never, Protocol, TypeVar, no_type_check

from yapf.yapflib.errors import YapfError

from ...loggers import logger
from ...patches.py_ import NonInstantiableMeta
from ..configs import Config


class Item(Protocol):
    __name__: str
    __qualname__: str

    def __call__(self, *args, **kwargs) -> Any:
        ...


T = TypeVar('T', bound=Item)

BuildPreHook = Callable[[Config, 'RegistryMeta', Any], Config]


class RegistryMeta(  # type: ignore[misc]
    UserDict[str, Item],
    NonInstantiableMeta,
):
    """Meta class for registries.

    Underneath, registries are simply dictionaries:

        >>> class Cat(metaclass=RegistryMeta): pass
        >>> class BritishShorthair: pass
        >>> Cat['british shorthair'] = BritishShorthair
        >>> Cat['british shorthair']
        <class '...BritishShorthair'>

    In this example, ``Cat`` is a registry and the "british shorthair" is a
    category in the registry.
    ``BritishShorthair`` is an object or class that is associated to the
    "british shorthair" category.

    For convenience, users can also access registries via higher level APIs,
    such as '`register_`' and '`build`'.
    These provide easier interfaces to register and retrieve instances:

        >>> class Persian: pass
        >>> Cat.register_('persian')(Persian)
        <class '...Persian'>
        >>> Cat.build(Config(type='persian'))
        <...Persian object at ...>

    Registries can be subclassed as well to create specializations or child
    registries:

        >>> class HairlessCat(Cat): pass
        >>> Cat.child('HairlessCat')
        <HairlessCat >

    In the example above, ``HairlessCat`` can be seen as a subcategory or
    specialization of ``Cat``.
    This allows to organize instances into a hierarchically structured
    registry.
    """

    def __init__(cls, *args, **kwargs) -> None:
        """Initialize."""
        UserDict.__init__(cls)
        NonInstantiableMeta.__init__(cls, *args, **kwargs)

    def __repr__(cls) -> str:
        items = ' '.join(f'{k}={v}' for k, v in cls.items())
        return f"<{cls.__name__} {items}>"

    def __hash__(cls) -> int:
        return id(cls)

    # Inheritance

    @no_type_check
    def __subclasses__(cls: Any = ...) -> Any:
        """Fetch subclasses of the current class.

        For more details, refer to `ABC subclassed by meta classes`_.

        .. _ABC subclassed by meta classes:
           https://blog.csdn.net/LutingWang/article/details/128320057
        """
        if cls is ...:
            return NonInstantiableMeta.__subclasses__(RegistryMeta)
        return super().__subclasses__()

    def child(cls, key: str) -> 'RegistryMeta':
        """Retrieve a direct or indirect derived child registry.

        Given a dot-separated string of subclass names, this method searches
        for the specified child registry within its inheritance tree and
        returns the matching child class.

        Args:
            key: A string of dot-separated subclass names.

        Raises:
            ValueError: If no subclass or more than one subclass with the
                specified name exists.

        Returns:
            The specified child registry.
        """
        child = cls
        for child_name in key.split('.'):
            subclasses = tuple(child.__subclasses__())  # type: ignore[misc]
            subclasses = tuple(
                subclass for subclass in subclasses
                if subclass.__name__ == child_name
            )
            if not subclasses:
                raise ValueError(f"{child_name} is not a child of {child}")
            if len(subclasses) > 1:
                raise ValueError(
                    f"{child_name} matches multiple children of {child}",
                )
            child, = subclasses
        return child

    def _parse(cls, key: str) -> tuple['RegistryMeta', str]:
        """Parse the ``key`` which may contain child classes separated by dots.

        Args:
            key: the string to be parsed.

        Returns:
            A tuple containing the child registry object and the updated key
            string.
        """
        if '.' not in key:
            return cls, key
        child_name, key = key.rsplit('.', 1)
        child = cls.child(child_name)
        return child, key

    # Retrieval

    def __missing__(cls, key: str) -> Never:
        """Missing key.

        Args:
            key: the missing key.

        Raises:
            KeyError: always.
        """
        logger.error("%s does not exist in %s", key, cls.__name__)
        raise KeyError(key)

    def parse(cls, key: str) -> tuple['RegistryMeta', Item]:
        """Parse ``key``.

        Returns:
            The child registry and the corresponding type.
        """
        child, key = cls._parse(key)
        item = super(RegistryMeta, child).__getitem__(key)
        return child, item

    def __contains__(cls, key) -> bool:
        if not isinstance(key, str):
            return False
        child, key = cls._parse(key)
        return super(RegistryMeta, child).__contains__(key)

    def __getitem__(cls, key: str) -> Item:
        _, item = cls.parse(key)
        return item

    # Registration

    def __setitem__(cls, key: str, item: Item) -> None:
        """Register ``item`` with name ``key``.

        Args:
            key: name to be registered as.
            item: object to be registered.

        Raises:
            KeyError: if ``key`` is already registered.

        `RegistryMeta` refuses to alter the registered object, in order to
        prevent unintended name clashes:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> Cat['british shorthair'] = 'british shorthair'
            >>> Cat['british shorthair'] = 'BritishShorthair'
            Traceback (most recent call last):
                ...
            KeyError: 'british shorthair'
        """
        if key in cls:  # noqa: E501 pylint: disable=unsupported-membership-test
            logger.error("%s already exist in %s", key, cls)
            raise KeyError(key)
        child, key = cls._parse(key)
        super(RegistryMeta, child).__setitem__(key, item)

    def register_(
        cls,
        *args: str,
        force: bool = False,
        build_pre_hook: BuildPreHook | None = None,
    ) -> Callable[[T], T]:
        """Register classes or functions to the registry.

        Args:
            args: names to be registered as.
            force: if set, registration will always happen.

        Returns:
            Wrapper function.

        The decorator can be applied to both classes and functions:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register_()
            ... class Munchkin: pass
            >>> @Cat.register_()
            ... def munchkin() -> str:
            ...     return 'munchkin'

        If no arguments are given, the name of the object being registered is
        used as the key:

            >>> Cat['Munchkin']
            <class '...Munchkin'>
            >>> Cat['munchkin']
            <function munchkin at ...>

        It is possible to register an object with multiple names:

            >>> @Cat.register_('British Longhair', 'british longhair')
            ... class BritishLonghair: pass
            >>> 'British Longhair' in Cat
            True
            >>> 'british longhair' in Cat
            True

        It also allows one to specify child registries as part of the key
        during registration:

            >>> class HairlessCat(Cat): pass
            >>> @Cat.register_('HairlessCat.CanadianHairless')
            ... def canadian_hairless() -> str:
            ...     return 'canadian hairless'
            >>> HairlessCat
            <HairlessCat CanadianHairless=<function canadian_hairless at ...>>

        If 'forced' is True and an item of the same name exists, the new item
        will replace the old one in the registry:

            >>> class AnotherMunchkin: pass
            >>> Cat.register_('Munchkin')(AnotherMunchkin)
            Traceback (most recent call last):
                ...
            KeyError: 'Munchkin'
            >>> Cat.register_('Munchkin', force=True)(AnotherMunchkin)
            <class '...AnotherMunchkin'>
            >>> Cat['Munchkin']
            <class '...AnotherMunchkin'>

        `build_pre_hook` can be bind to objects during registration:

            >>> build_pre_hook = lambda c, r, i: c
            >>> @Cat.register_(build_pre_hook=build_pre_hook)
            ... class Maine: pass
            >>> Maine.build_pre_hook is build_pre_hook
            True
        """

        def wrapper_func(item: T) -> T:
            keys = args or [item.__name__]
            for key in keys:
                if force:
                    cls.pop(key, None)
                cls[key] = item  # noqa: E501 pylint: disable=unsupported-assignment-operation
            if build_pre_hook is not None:
                item.build_pre_hook = build_pre_hook  # type: ignore[attr-defined] # noqa: E501 pylint: disable=line-too-long
            return item

        return wrapper_func

    # Deregistration

    def __delitem__(cls, key: str) -> None:
        child, key = cls._parse(key)
        return super(RegistryMeta, child).__delitem__(key)

    # Construction

    def _build(cls, item: Item, config: Config) -> Any:
        """Build an instance according to the given config.

        Args:
            item: instance type.
            config: instance specification.

        Returns:
            The built instance.

        To customize the build process of instances, registries must overload
        `_build` with a class method:

            >>> class Cat(metaclass=RegistryMeta):
            ...     @classmethod
            ...     def _build(cls, item: Item, config: Config):
            ...         obj = RegistryMeta._build(cls, item, config)
            ...         obj.name = obj.name.upper()
            ...         return obj
            >>> @Cat.register_()
            ... class Munchkin:
            ...     def __init__(self, name: str) -> None:
            ...         self.name = name
            >>> config = Config(type='Munchkin', name='Garfield')
            >>> cat = Cat.build(config)
            >>> cat.name
            'GARFIELD'
        """
        return item(**config)

    def build(cls, config: Config, **kwargs) -> Any:
        """Call the registered object to construct a new instance.

        Args:
            config: build parameters.
            kwargs: default configuration.

        Returns:
            The built instance.

        The ``type`` entry of ``config`` specifies the name of the registered
        object to be built.
        The other entries of ``config`` will be passed to the object's call
        method.

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register_()
            ... def tabby(name: str) -> str:
            ...     return f'Tabby {name}'
            >>> Cat.build(Config(type='tabby', name='Garfield'))
            'Tabby Garfield'

        Keyword arguments are the default configuration:

            >>> Cat.build(
            ...     Config(type='tabby'),
            ...     name='Garfield',
            ... )
            'Tabby Garfield'

        Override :meth:`_build` for customization:

            >>> class DomesticCat(Cat):
            ...     @classmethod
            ...     def _build(cls, item: Item, config: Config):
            ...         return item, config
            >>> @DomesticCat.register_()
            ... class Maine: pass
            >>> DomesticCat.build(Config(type='Maine', name='maine'), age=1.2)
            (<class '...Maine'>, {'age': 1.2, 'name': 'maine'})

        If the object has a property named ``build_pre_hook``, the config is
        converted before construction:

            >>> @Cat.register_()
            ... class Persian:
            ...     def __init__(self, friend: str) -> None:
            ...         self.friend = friend
            ...     @classmethod
            ...     def build_pre_hook(
            ...         cls,
            ...         config: Config,
            ...         registry: RegistryMeta,
            ...         item: Item,
            ...     ) -> Config:
            ...         config.friend = config.friend.type
            ...         return config
            >>> persian = Cat.build(
            ...     Config(type='Persian'),
            ...     friend=dict(type='Siamese'),
            ... )
            >>> persian.friend
            'Siamese'
        """
        # NOTE: If `kwargs` and `config` have the same key, the following code
        # will overwrite `kwargs` with `config`, instead of merging them.
        # config = Config(kwargs) | config

        kwargs = Config(kwargs)
        kwargs.update(config)
        config = kwargs

        config_type = config.pop('type')
        registry, item = cls.parse(config_type)

        build_pre_hook: BuildPreHook | None = getattr(
            item,
            'build_pre_hook',
            None,
        )
        if build_pre_hook is not None:
            try:
                config = build_pre_hook(config.copy(), registry, item)
            except Exception:
                from ...configs import PyConfig
                try:
                    dumps = PyConfig(config).dumps()
                except YapfError:
                    dumps = repr(config)
                logger.error(
                    "Failed to preprocess %s:\n%s",
                    config_type,
                    dumps,
                )
                raise

        try:
            return registry._build(item, config.copy())
        except Exception:
            from ...configs import PyConfig
            try:
                dumps = PyConfig(config).dumps()
            except YapfError:
                dumps = repr(config)
            logger.error("Failed to build %s:\n%s", config_type, dumps)
            raise

    def build_or_return(
        cls,
        config: Any,
        predicate: Callable[[Any], bool] | None = None,
        **kwargs,
    ) -> Any:
        build = (
            isinstance(config, Config)
            if predicate is None else predicate(config)
        )
        return cls.build(config, **kwargs) if build else config


class Registry(metaclass=RegistryMeta):
    """Base registry.

    To create custom registry, inherit from the `Registry` class:

        >>> class CatRegistry(Registry): pass
    """
