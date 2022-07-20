__all__ = [
    'Message',
    'STEPS',
    'WorkflowConfig',
    'Workflow',
]

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Union,
)

from ._extensions import Config, get_logger
from .registries import Registry

Message = Dict[str, Any]

Step = Callable[..., Any]
STEPS: Registry[Step] = Registry('steps')


class BaseStepManager:

    @abstractmethod
    def __iter__(self) -> Generator[Any, None, None]:
        pass

    @abstractmethod
    def __call__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        pass

    @property
    @abstractmethod
    def step(self):
        pass


class SingleStepManager(BaseStepManager):

    def __init__(self, step: Step) -> None:
        self._step = step

    def __iter__(self) -> Generator[Any, None, None]:
        yield self._step

    def __call__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        return self._step(*args, **kwargs)

    @property
    def step(self) -> Step:
        return self._step


class ParallelSingleStepManager(SingleStepManager):

    def __call__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        super_call = super().__call__
        return tuple(super_call(p_args, kwargs) for p_args in zip(*args))


class ParallelMultiStepManager(BaseStepManager):

    def __init__(self, steps: Iterable[Step]) -> None:
        self._steps = tuple(steps)

    def __iter__(self) -> Generator[Any, None, None]:
        yield from self._steps

    def __call__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        return tuple(  # yapf: disable
            step(*p_args, **kwargs)
            for step, *p_args in zip(self._steps, *args)
        )

    @property
    def step(self) -> Tuple[Step, ...]:
        return self._steps


class BaseOutputDict:

    @abstractmethod
    def __call__(self, output) -> Dict[str, Any]:
        pass


class SingleOutputDict(BaseOutputDict):

    def __init__(self, key: str) -> None:
        self._key = key

    def __call__(self, data) -> Dict[str, Any]:
        return {self._key: data}


class MultiOutputDict(BaseOutputDict):

    def __init__(self, keys: Iterable[str]) -> None:
        self._keys = tuple(keys)

    def __call__(self, output) -> Dict[str, Any]:
        return dict(zip(self._keys, output))


class ParallelMultiOutputDict(MultiOutputDict):

    def __call__(self, output) -> Dict[str, Tuple[Any, ...]]:
        output = zip(*output)
        return dict(zip(self._keys, output))


OutputKey = Union[str, Tuple[str, ...]]


class Job:

    @staticmethod
    def build_step_manager(
        config: Config,
        registry: Registry[Step],
        parallel,
    ) -> BaseStepManager:
        if isinstance(parallel, bool):
            step = registry.build(config)
            if parallel:
                return ParallelSingleStepManager(step)
            return SingleStepManager(step)
        if isinstance(parallel, int):
            steps = tuple(registry.build(config) for _ in range(parallel))
            return ParallelMultiStepManager(steps)
        if isinstance(parallel, Iterable):
            steps = tuple(
                registry.build(p_config, config) for p_config in parallel
            )
            return ParallelMultiStepManager(steps)
        raise TypeError(
            "`parallel` must be a bool, int, or Iterable, "
            f"but got {type(parallel)}"
        )

    @staticmethod
    def build_output_dict(key: OutputKey, parallel) -> BaseOutputDict:
        if isinstance(key, str):
            return SingleOutputDict(key)
        if parallel:
            return ParallelMultiOutputDict(key)
        return MultiOutputDict(key)

    @staticmethod
    def build(
        step_descendent_name: str,
        config: Config,
        output_key: OutputKey,
    ) -> 'Job':
        registry = STEPS.descendent(step_descendent_name)
        parallel = config.pop('parallel', False)
        fields = config.pop('fields', tuple())
        step_manager = Job.build_step_manager(config, registry, parallel)
        output_dict = Job.build_output_dict(output_key, parallel)
        return Job(step_manager, output_dict, fields)

    def __init__(
        self,
        step_manager: BaseStepManager,
        output_dict: BaseOutputDict,
        fields: Iterable[str] = tuple(),
    ) -> None:
        self._step_manager = step_manager
        self._output_dict = output_dict
        self._fields = tuple(fields)
        self._logger = get_logger()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._step_manager)

    def __call__(self, message: Message, **kwargs) -> Message:
        inputs = tuple(message[field] for field in self._fields)
        try:
            output = self._step_manager(inputs, kwargs)
        except Exception:
            self._logger.error(f"Failed to forward {self}")
            raise
        return self._output_dict(output)

    @property
    def step(self):
        return self._step_manager.step


WorkflowConfig = Dict[OutputKey, Config]


class Workflow:

    @staticmethod
    def build(
        step_descendent_name: str,
        configs: Optional[WorkflowConfig] = None,
        **kwargs: Config,
    ) -> 'Workflow':
        jobs = tuple(  # yapf: disable
            Job.build(step_descendent_name, config, output_key)
            for output_key, config in (configs or kwargs).items()
        )
        return Workflow(jobs)

    def __init__(self, jobs: Iterable[Job]) -> None:
        self._jobs = tuple(jobs)

    def __iter__(self) -> Iterator[Job]:
        return iter(self._jobs)

    def __call__(self, message: Message) -> Message:
        updated: Message = dict()
        for job in self._jobs:
            updates = job(message)
            message.update(updates)
            updated.update(updates)
        return updated
