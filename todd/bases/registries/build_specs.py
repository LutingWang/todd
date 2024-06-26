__all__ = [
    'BuildSpec',
    'BuildSpecMixin',
]

from collections import Counter, UserDict
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping

from ...patches.py import classproperty
from .builders import BaseBuilder, Builder

if TYPE_CHECKING:
    from ...configs import Config

F = Callable[..., Any]


class BuildSpec(UserDict[str, BaseBuilder | F]):
    """A class representing a build specification.

    A build specification is a mapping from keys to functions:

        >>> build_spec = BuildSpec(age=lambda c: c.value)
        >>> build_spec
        {'age': <function ...>}

    If a key of the build specification appears in the given configuration and
    the corresponding value is a `Config` object, the function will be applied
    to the value:

        >>> from todd import Config
        >>> config = Config(age=Config(value=3))
        >>> build_spec(config)
        {'age': 3}

    If the corresponding value is not a `Config` object, the function will not
    be applied:

        >>> config = Config(age='4')
        >>> build_spec(config)
        {'age': '4'}

    Under the hood, the functions are wrapped in `Builder` objects.
    The above example is equivalent to:

        >>> build_spec = BuildSpec(age=Builder(lambda c: c.value))
        >>> build_spec
        {'age': <...Builder object at ...>}
        >>> config = Config(age=Config(value=3))
        >>> build_spec(config)
        {'age': 3}

    Users can initialize other builders to control the build process:

        >>> from todd.bases.registries import NestedCollectionBuilder
        >>> builder = NestedCollectionBuilder(lambda c: c.name)
        >>> build_spec = BuildSpec(friends=builder)
        >>> build_spec
        {'friends': <...NestedCollectionBuilder object at ...>}
        >>> config = Config(friends=[Config(name='Alice'), Config(name='Bob')])
        >>> build_spec(config)
        {'friends': ('Alice', 'Bob')}

    `BaseBuilder` objects can depend on each other:

        >>> dataset_builder = Builder(lambda c: c.name)
        >>> dataloader_builder = Builder(
        ...     lambda c, ds: f'{c.type_} with {ds}',
        ...     requires=dict(dataset='ds'),
        ... )
        >>> build_spec = BuildSpec(
        ...     dataset=dataset_builder,
        ...     dataloader=dataloader_builder,
        ... )
        >>> config = Config(
        ...     dataloader=dict(type_='Dataloader'),
        ...     dataset=dict(name='MNIST'),
        ... )
        >>> result = build_spec(config)
        >>> dict(sorted(result.items()))
        {'dataloader': 'Dataloader with MNIST', 'dataset': 'MNIST'}
    """

    def sort(
        self,
        graph: Mapping[str, set[str]],
    ) -> Generator[str, None, None]:
        in_degrees = Counter({k: len(v) for k, v in graph.items()})
        queue = {k for k, v in in_degrees.items() if v == 0}
        while queue:
            node = queue.pop()
            yield node
            out_nodes = {k for k, v in graph.items() if node in v}
            in_degrees.subtract(out_nodes)
            queue = queue.union(k for k in out_nodes if in_degrees[k] == 0)

    def __call__(self, config: 'Config') -> 'Config':
        builders: dict[str, BaseBuilder] = dict()
        for k, builder in self.items():
            if k not in config:
                continue
            v = config[k]
            if not isinstance(builder, BaseBuilder):
                builder = Builder(builder)
            if not builder.should_build(v):
                continue
            builders[k] = builder

        graph = {
            k: builders.keys() & builder.requires
            for k, builder in builders.items()
        }
        for k in self.sort(graph):
            builder = builders.pop(k)
            v = config[k]
            requires = {
                require: config[require]
                for require in builder.requires
            }
            config[k] = builder(v, **requires)

        if builders:
            raise ValueError(f"Unresolved dependencies: {builders.keys()}")
        return config


class BuildSpecMixin:

    @classproperty
    def build_spec(self) -> BuildSpec:
        return BuildSpec()
