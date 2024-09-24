__all__ = [
    'Message',
    'Spec',
    'Processor',
    'Operator',
    'SingleMixin',
    'MultipleMixin',
    'SingleOperator',
    'ParallelMixin',
    'SingleParallelOperator',
    'MultipleParallelOperator',
    'Pipeline',
]

from abc import ABC, abstractmethod
from symtable import symtable
from typing import Any, Callable, Generic, Iterable, NamedTuple, TypeVar, cast

import pandas as pd

from todd import Config
from todd.bases.registries import (
    BuildPreHookMixin,
    Item,
    Registry,
    RegistryMeta,
)
from todd.loggers import logger
from todd.patches.py_ import exec_
from todd.utils import Args, ArgsKwargs, Kwargs, SerializeMixin

from ..registries import KDProcessorRegistry

Message = dict[str, Any]

T_co = TypeVar('T_co', bound=Callable, covariant=True)


class Spec(NamedTuple):
    inputs: set[str]
    outputs: set[str]


class Processor(BuildPreHookMixin, SerializeMixin, ABC, Generic[T_co]):

    @abstractmethod
    def __call__(self, message: Message) -> Message:
        """Execute the processor.

        Args:
            message: inputs.

        Returns:
            Outputs.
        """

    @property
    @abstractmethod
    def spec(self) -> Spec:
        """Return the specification of the processor."""

    @property
    @abstractmethod
    def atoms(self) -> tuple[T_co, ...]:
        """User-defined callables used by the processor."""


class Operator(Processor[T_co], ABC):

    def __init__(
        self,
        *args_,
        args: Iterable[str],
        kwargs: Config | None = None,
        outputs: str,
        **kwargs_,
    ) -> None:
        """Initialize.

        Args:
            args: names of the input fields.
            kwargs: name mapping of the input fields.
            outputs: expression of the outputs.

        For convenience, ``outputs`` is designed to be an expression.
        Suppose ``outputs`` is ``a, b``, the behavior of the processor is
        similar to the following code:

        .. code-block:: python

           a, b = callable_(...)
           return dict(a=a, b=b)
        """
        super().__init__(*args_, **kwargs_)
        self._args = tuple(args)
        self._kwargs = Config(kwargs)
        self._outputs = outputs

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        kwargs.update(
            args=self._args,
            kwargs=self._kwargs,
            outputs=self._outputs,
        )
        return args, kwargs

    def __call__(self, message: Message) -> Message:
        """Execute the processor.

        Args:
            message: inputs.

        Returns:
            Outputs.
        """
        args = tuple(message[input_] for input_ in self._args)
        kwargs = {k: message[v] for k, v in self._kwargs.items()}
        return self._operate(args, kwargs)

    @property
    def spec(self) -> Spec:
        return Spec(
            set(self._args) | set(self._kwargs.values()),
            set(symtable(self._outputs, str(self), 'eval').get_identifiers()),
        )

    def _parse_outputs(self, outputs: Any) -> Message:
        return exec_(f'{self._outputs} = __o', __o=outputs)

    @abstractmethod
    def _operate(self, args: Args, kwargs: Kwargs) -> Message:
        pass


class SingleMixin(Operator[T_co], ABC):

    def __init__(
        self,
        *args,
        atom: T_co,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._atom = atom

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config.atom = Registry.build_or_return(config.atom)
        return config

    @property
    def atoms(self) -> tuple[T_co, ...]:
        return (self._atom, )

    def _single_operate(self, args: Args, kwargs: Kwargs) -> Message:
        outputs = self._atom(*args, **kwargs)
        return self._parse_outputs(outputs)


class MultipleMixin(Operator[T_co], ABC):

    def __init__(
        self,
        *args,
        atoms: Iterable[T_co],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._atoms = tuple(atoms)

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config.atoms = [Registry.build_or_return(c) for c in config.atoms]
        return config

    @property
    def atoms(self) -> tuple[T_co, ...]:
        return self._atoms

    def _multiple_operate(self, i: int, args: Args, kwargs: Kwargs) -> Message:
        outputs = self._atoms[i](*args, **kwargs)
        return self._parse_outputs(outputs)


@KDProcessorRegistry.register_()
class SingleOperator(SingleMixin[T_co], Operator[T_co]):

    def _operate(self, args: Args, kwargs: Kwargs) -> Message:
        return self._single_operate(args, kwargs)


class ParallelMixin(Operator[T_co], ABC):

    @abstractmethod
    def _parallel_operate(self, i: int, args: Args, kwargs: Kwargs) -> Message:
        pass

    def _operate(self, args: Args, kwargs: Kwargs) -> Message:
        n, = set(map(len, args)) | set(map(len, kwargs.values()))
        # yapf: disable
        outputs = [
            self._parallel_operate(
                i,
                tuple(a[i] for a in args),
                {k: v[i] for k, v in kwargs.items()},
            ) for i in range(n)
        ]
        # yapf: enable
        return cast(Message, pd.DataFrame(outputs).to_dict(orient='list'))


@KDProcessorRegistry.register_()
class SingleParallelOperator(
    SingleMixin[T_co],
    ParallelMixin[T_co],
    Operator[T_co],
):

    def _parallel_operate(self, i: int, args: Args, kwargs: Kwargs) -> Message:
        return self._single_operate(args, kwargs)


@KDProcessorRegistry.register_()
class MultipleParallelOperator(
    MultipleMixin[T_co],
    ParallelMixin[T_co],
    Operator[T_co],
):

    def _parallel_operate(self, i: int, args: Args, kwargs: Kwargs) -> Message:
        return self._multiple_operate(i, args, kwargs)


@KDProcessorRegistry.register_()
class Pipeline(Processor[T_co]):

    def __init__(
        self,
        *args,
        processors: Iterable[Processor[T_co]],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processors = tuple(processors)

    @classmethod
    def processors_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        processors = config.processors
        if isinstance(processors, Config):
            processors = [
                KDProcessorRegistry.build_or_return(v, outputs=k)
                for k, v in processors.items()
            ]
        else:
            processors = [
                KDProcessorRegistry.build_or_return(c) for c in processors
            ]
        config.processors = [p for p in processors if p is not None]
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.processors_build_pre_hook(config, registry, item)
        return config

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        kwargs['processors'] = self._processors
        return args, kwargs

    def __call__(self, message: Message) -> Message:
        updates: Message = dict()
        for processor in self._processors:
            try:
                message_ = processor(message)
            except Exception:
                logger.error("Failed to forward %s", processor)
                raise
            message.update(message_)
            updates.update(message_)
        return updates

    @property
    def spec(self) -> Spec:
        inputs: set[str] = set()
        outputs: set[str] = set()
        for processor in self._processors:
            spec = processor.spec
            inputs |= spec.inputs - outputs
            outputs |= spec.outputs
        return Spec(inputs, outputs)

    @property
    def atoms(self) -> tuple[T_co, ...]:
        return sum(
            (processor.atoms for processor in self._processors),
            tuple(),
        )

    @property
    def processors(self) -> tuple[Processor[T_co], ...]:
        return self._processors
