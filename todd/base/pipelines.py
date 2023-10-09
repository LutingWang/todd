__all__ = [
    'Spec',
    'BasePipeline',
    'IOPipeline',
    'VanillaPipeline',
    'ParallelPipeline',
    'MultipleParallelPipeline',
    'SingleParallelPipeline',
    'ComposedPipeline',
]

import itertools
from abc import ABC, abstractmethod
from symtable import symtable
from typing import (
    TYPE_CHECKING,
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

from ..utils import exec_
from .configs import Config
from .logger import logger
from .registries import PipelineRegistry, RegistryMeta

Message = dict[str, Any]
Pipelines = Iterable[Config] | Mapping[str, Config]

T = TypeVar('T', bound=Callable)


class Spec(NamedTuple):
    inputs: set[str]
    outputs: set[str]


class BasePipeline(Generic[T], ABC):

    def __init__(self, callable_registry: RegistryMeta) -> None:
        self._callable_registry = callable_registry

    @abstractmethod
    def __call__(self, message: Message) -> Message:
        """Executes the pipeline.

        Args:
            message: inputs.

        Returns:
            Outputs.
        """
        pass

    @property
    @abstractmethod
    def callables(self) -> tuple[T, ...]:
        """User-defined callables used by the pipeline."""
        pass

    @property
    @abstractmethod
    def spec(self) -> Spec:
        """Specifications of the pipeline."""
        pass

    def build_callable(self, config: Config) -> T:
        return self._callable_registry.build(config)


class IOPipeline(BasePipeline[T]):

    def __init__(
        self,
        *args,
        inputs: Iterable[str],
        outputs: str,
        **kwargs,
    ) -> None:
        """Initialize.

        Args:
            inputs: names of the input fields.
            outputs: expression of the outputs.

        For convenience, ``outputs`` is designed to be an expression.
        Suppose ``outputs`` is ``a, b``, the behavior of the pipeline is
        similar to the following code:

        .. code-block:: python

           a, b = callable_(...)
           return dict(a=a, b=b)
        """
        super().__init__(*args, **kwargs)
        self._inputs = tuple(inputs)
        self._outputs = outputs

    @property
    def spec(self) -> Spec:
        return Spec(
            set(self._inputs),
            set(symtable(self._outputs, '<string>', 'eval').get_identifiers()),
        )

    def inputs(self, message: Message) -> tuple:
        """Parse the inputs.

        Args:
            message: the original message.

        Returns:
            The parsed inputs.
        """
        return tuple(message[input_] for input_ in self._inputs)

    def outputs(self, outputs) -> Message:
        """Parse the outputs.

        Args:
            outputs: outputs of the action.

        Returns:
            The parsed outputs.
        """
        message = exec_(f'{self._outputs} = __o', __o=outputs)
        return message


@PipelineRegistry.register()
class VanillaPipeline(IOPipeline[T]):

    def __init__(
        self,
        *args,
        callable_: Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._build_callable(callable_)

    def _build_callable(self, config: Config) -> None:
        self._callable = self.build_callable(config)

    def __call__(self, message: Message) -> Message:
        inputs = self.inputs(message)
        outputs = self._callable(*inputs)
        return self.outputs(outputs)

    @property
    def callables(self) -> tuple[T, ...]:
        return self._callable,


class ParallelPipeline(IOPipeline[T]):
    _callables: Iterable[T]

    def __call__(self, message: Message) -> Message:
        inputs = self.inputs(message)
        messages = []
        for callable_, *inputs_ in zip(self._callables, *inputs):
            outputs = callable_(*inputs_)
            messages.append(self.outputs(outputs))
        data_frame = pd.DataFrame(messages)
        return cast(Message, data_frame.to_dict(orient="list"))


@PipelineRegistry.register()
class MultipleParallelPipeline(ParallelPipeline[T]):
    _callables: tuple[T, ...]

    def __init__(self, *args, callables: Iterable[Config], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._build_callables(callables)

    def _build_callables(self, config: Iterable[Config]) -> None:
        self._callables = tuple(map(self.build_callable, config))

    @property
    def callables(self) -> tuple[T, ...]:
        return self._callables


@PipelineRegistry.register()
class SingleParallelPipeline(ParallelPipeline[T]):
    if TYPE_CHECKING:
        _callables: itertools.repeat[T]

    def __init__(self, *args, callable_: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._build_callable(callable_)

    def _build_callable(self, config: Config) -> None:
        callable_ = self.build_callable(config)
        self._callables = itertools.repeat(callable_)

    @property
    def callables(self) -> tuple[T, ...]:
        return next(self._callables),


@PipelineRegistry.register()
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
                logger.error(f"Failed to forward {pipeline}")
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
