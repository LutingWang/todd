from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch.nn as nn

from ._extensions import ModuleList, SequenceProto, get_logger
from .misc import strict_zip_len
from .registries import Registry

__all__ = [
    'STEPS',
    'Step',
    'Job',
    'ModuleStep',
    'ModuleJob',
]

StepType = TypeVar('StepType', bound='Step')
JobType = TypeVar('JobType', bound='Job')


class Step:
    REGISTRY: Registry

    @classmethod
    def build(
        cls: Type[StepType],
        cfg: Union[dict, StepType],
        default_args: Optional[dict] = None,
    ) -> StepType:
        if isinstance(cfg, cls):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError(
                f'Step config must be a dict or an instance of {cls.__name__}'
            )
        if default_args is not None:
            for k, v in default_args.items():
                cfg.setdefault(k, v)
        return cls(**cfg)

    def __init__(
        self,
        id_: Union[str, Iterable[str]],
        fields: Iterable[str],
        parallel: Union[bool, int, Iterable[dict]] = False,
        **default_args,
    ) -> None:
        self._id = id_
        self._fields = tuple(fields)
        self._parallel = parallel
        self._logger = get_logger()

        self._executor: Callable[..., Any]
        self._executors: SequenceProto[Callable[..., Any]]
        if isinstance(parallel, bool):
            self._executor = self.REGISTRY.build(default_args)
        elif isinstance(parallel, int):
            self._executors = tuple(
                self.REGISTRY.build(default_args) for _ in range(parallel)
            )
        elif isinstance(parallel, Iterable):
            self._executors = tuple(
                self.REGISTRY.build(kwargs, default_args)
                for kwargs in parallel
            )
        else:
            raise TypeError(
                "`parallel` must be a bool, int, or Iterable, "
                f"but got {type(parallel)}"
            )

    def _forward(self, inputs: tuple, kwargs: dict):
        if isinstance(self._parallel, bool):
            if not self._parallel:
                return self._executor(*inputs, **kwargs)
            return tuple(
                self._executor(*parallel_inputs, **kwargs)
                for parallel_inputs in zip(*inputs)
            )
        return tuple(  # yapf: disable
            executor(*parallel_inputs, **kwargs)
            for executor, *parallel_inputs in zip(self._executors, *inputs)
        )

    def forward(self, message: dict, **kwargs) -> dict:
        inputs = tuple(message[field] for field in self._fields)
        if not isinstance(self._parallel, bool):
            input_len = strict_zip_len(inputs)
            if len(self._executors) != input_len:
                raise ValueError(
                    "Lengths of `inputs` and `self._executer` must be equal, "
                    f"but got input_len={input_len} and len(self._executer)="
                    f"{len(self._executors)}"
                )

        try:
            outputs = self._forward(inputs, kwargs)
        except Exception:
            self._logger.error(f"Failed to forward {self._id}")
            raise

        if isinstance(self._id, str):
            return {self._id: outputs}
        else:
            return dict(zip(self._id, outputs))


STEPS = Registry('steps', base=Step)
StepCfg = Union[dict, Step]


class Job:
    STEP_TYPE = 'Step'

    @classmethod
    def build(
        cls: Type[JobType],
        cfg: Union[Dict[str, StepCfg], Iterable[StepCfg], JobType],
        *args,
        **kwargs,
    ) -> JobType:
        if isinstance(cfg, cls):
            return cfg
        cfg = cast(Union[Dict[str, StepCfg], Iterable[StepCfg]], cfg)
        return cls(cfg, *args, **kwargs)

    def __init__(
        self,
        steps: Union[Dict[str, StepCfg], Iterable[StepCfg]],
    ) -> None:
        self._steps: SequenceProto[Step]
        step_type = STEPS[self.STEP_TYPE]
        if isinstance(steps, dict):
            self._steps = tuple(  # yapf: disable
                step_type.build(
                    step,
                    default_args=dict(id_=id_),
                )
                for id_, step in steps.items()
            )
        elif isinstance(steps, Iterable):
            steps = cast(Iterable[StepCfg], steps)
            self._steps = tuple(step_type.build(step) for step in steps)
        else:
            raise TypeError(
                "`steps` must be a dict or Iterable, "
                f"but got steps={steps}"
            )

    def forward(self, message: dict, inplace: bool = False) -> dict:
        if not inplace:
            message = dict(message)
        updated_message = dict()
        for step in self._steps:
            updates = step.forward(message)
            message.update(updates)
            updated_message.update(updates)
        return updated_message


@STEPS.register_module()
class ModuleStep(Step, nn.Module):

    def __init__(
        self,
        id_: Union[str, Iterable[str]],
        tensor_names: Iterable[str],
        multilevel: Union[bool, int, Iterable[dict]] = False,
        **default_args,
    ) -> None:
        nn.Module.__init__(self)
        Step.__init__(
            self,
            id_=id_,
            fields=tensor_names,
            parallel=multilevel,
            **default_args,
        )
        if isinstance(self._parallel, bool):
            return
        self._executors = ModuleList(self._executors)


class ModuleJob(Job, nn.Module):
    STEP_TYPE = 'ModuleStep'

    def __init__(
        self,
        steps: Union[Dict[str, StepCfg], Iterable[StepCfg]],
    ) -> None:
        nn.Module.__init__(self)
        Job.__init__(self, steps)
        self._steps = ModuleList(self._steps)
