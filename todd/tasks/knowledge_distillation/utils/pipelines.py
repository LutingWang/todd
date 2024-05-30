__all__ = [
    'Spec',
    'BasePipeline',
    'IOMixin',
    'VanillaPipeline',
    'ParallelIOMixin',
    'ParallelPipeline',
    'SharedParallelPipeline',
    'ComposedPipeline',
]

import itertools
import operator
from abc import ABC, abstractmethod
from symtable import symtable
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    TypeVar,
    cast,
)

import pandas as pd

from ....configs import Config
from ....loggers import logger
from ....patches.py import exec_
from ....registries import RegistryMeta
from ..registries import PipelineRegistry

# TODO: refactor

Message = dict[str, Any]
Pipelines = Iterable[Config] | Mapping[str, Config]
Args = tuple
Kwargs = dict[str, Any]

T = TypeVar('T', bound=Callable)


class Spec(NamedTuple):
    inputs: set[str]
    outputs: set[str]


class BasePipeline(Generic[T], ABC):

    def __init__(self, callable_registry: RegistryMeta) -> None:
        self._callable_registry = callable_registry

    @abstractmethod
    def __call__(self, message: Message) -> Message:
        """Execute the pipeline.

        Args:
            message: inputs.

        Returns:
            Outputs.
        """

    @property
    @abstractmethod
    def callables(self) -> tuple[T, ...]:
        """User-defined callables used by the pipeline."""

    @property
    @abstractmethod
    def spec(self) -> Spec:
        """Specifications of the pipeline."""


class IOMixin(BasePipeline[T]):

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
        Suppose ``outputs`` is ``a, b``, the behavior of the pipeline is
        similar to the following code:

        .. code-block:: python

           a, b = callable_(...)
           return dict(a=a, b=b)
        """
        super().__init__(*args_, **kwargs_)
        self._args = tuple(args)
        self._kwargs = Config() if kwargs is None else Config(kwargs)
        self._outputs = outputs

    @property
    def spec(self) -> Spec:
        return Spec(
            set(self._args) | set(self._kwargs.values()),
            set(symtable(self._outputs, '<string>', 'eval').get_identifiers()),
        )

    def args(self, message: Message) -> Args:
        """Parse the inputs.

        Args:
            message: the original message.

        Returns:
            The parsed inputs.
        """
        return tuple(message[input_] for input_ in self._args)

    def kwargs(self, message: Message) -> Kwargs:
        """Parse the inputs.

        Args:
            message: the original message.

        Returns:
            The parsed inputs.
        """
        return {k: message[v] for k, v in self._kwargs.items()}

    def outputs(self, outputs) -> Message:
        """Parse the outputs.

        Args:
            outputs: outputs of the action.

        Returns:
            The parsed outputs.
        """
        message = exec_(f'{self._outputs} = __o', __o=outputs)
        return message

    @abstractmethod
    def _call(self, args: Args, kwargs: Kwargs):
        pass

    def __call__(self, message: Message) -> Message:
        args = self.args(message)
        kwargs = self.kwargs(message)
        outputs = self._call(args, kwargs)
        return self.outputs(outputs)


@PipelineRegistry.register_()
class VanillaPipeline(IOMixin[T], BasePipeline[T]):

    def __init__(
        self,
        *args,
        callable_: Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._build_callable(callable_)

    def _build_callable(self, config: Config) -> None:
        self._callable = self._callable_registry.build(config)

    def _call(self, args: Args, kwargs: Kwargs):
        return self._callable(*args, **kwargs)

    @property
    def callables(self) -> tuple[T, ...]:
        return (self._callable, )


class ParallelIOMixin(IOMixin[T]):

    @abstractmethod
    def _parallelism(self, args: Args, kwargs: Kwargs) -> int:
        pass

    def _single_args(self, args: Args, i: int) -> Args:
        return tuple(map(operator.itemgetter(i), args))

    def _single_kwargs(self, kwargs: Kwargs, i: int) -> Kwargs:
        return {k: v[i] for k, v in kwargs.items()}

    def _single_outputs(self, outputs) -> Message:
        return super().outputs(outputs)

    @abstractmethod
    def _single_call(self, args: Args, kwargs: Kwargs, i: int):
        pass

    def _call(self, args: Args, kwargs: Kwargs) -> tuple[Message, ...]:
        parallelism = self._parallelism(args, kwargs)
        outputs = []
        for i in range(parallelism):
            single_args = self._single_args(args, i)
            single_kwargs = self._single_kwargs(kwargs, i)
            single_outputs = self._single_call(single_args, single_kwargs, i)
            outputs.append(self._single_outputs(single_outputs))
        return tuple(outputs)

    def outputs(self, outputs) -> Message:
        outputs = cast(tuple[Message, ...], outputs)
        data_frame = pd.DataFrame(outputs)
        return cast(Message, data_frame.to_dict(orient='list'))


@PipelineRegistry.register_()
class ParallelPipeline(ParallelIOMixin[T], BasePipeline[T]):

    def __init__(self, *args, callables: Iterable[Config], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._build_callables(callables)

    def _build_callables(self, config: Iterable[Config]) -> None:
        self._callables = tuple(map(self._callable_registry.build, config))

    @property
    def callables(self) -> tuple[T, ...]:
        return self._callables

    def _parallelism(self, args: Args, kwargs: Kwargs) -> int:
        return len(self._callables)

    def _single_call(self, args: Args, kwargs: Kwargs, i: int):
        return self._callables[i](*args, **kwargs)


@PipelineRegistry.register_()
class SharedParallelPipeline(ParallelIOMixin[T], VanillaPipeline[T]):

    def _parallelism(self, args: Args, kwargs: Kwargs) -> int:
        parallelism = set(map(len, itertools.chain(args, kwargs.values())))
        assert len(parallelism) == 1
        return parallelism.pop()

    def _single_call(self, args: Args, kwargs: Kwargs, i: int):
        return VanillaPipeline._call(self, args, kwargs)


@PipelineRegistry.register_()
class ComposedPipeline(BasePipeline[T]):

    def __init__(self, *args, pipelines: Pipelines, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(pipelines, Mapping):
            pipelines = [Config(outputs=k, **v) for k, v in pipelines.items()]
        self._build_pipelines(pipelines)

    def _build_pipelines(self, config: Iterable[Config]) -> None:
        self._pipelines: tuple[BasePipeline[T], ...] = tuple(
            PipelineRegistry.build(
                pipeline,
                callable_registry=self._callable_registry,
            ) for pipeline in config
        )

    def __call__(self, message: Message) -> Message:
        updates: Message = dict()
        for pipeline in self._pipelines:
            try:
                message_ = pipeline(message)
            except Exception:
                logger.error("Failed to forward %s", pipeline)
                raise
            message.update(message_)
            updates.update(message_)
        return updates

    @property
    def callables(self) -> tuple[T, ...]:
        return sum(
            (pipeline.callables for pipeline in self._pipelines),
            tuple(),
        )

    @property
    def pipelines(self) -> tuple[BasePipeline[T], ...]:
        return self._pipelines

    @property
    def spec(self) -> Spec:
        inputs: set[str] = set()
        outputs: set[str] = set()
        for pipeline in self._pipelines:
            spec = pipeline.spec
            inputs |= spec.inputs - outputs
            outputs |= spec.outputs
        return Spec(inputs, outputs)
