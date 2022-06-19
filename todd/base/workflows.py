from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch.nn as nn

from ._extensions import get_logger
from .misc import strict_zip_len
from .registries import Registry

__all__ = [
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
        cfg = cast(dict, cfg)
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

        executer: Any
        if isinstance(parallel, bool):
            executer = self.REGISTRY.build(default_args)
        elif isinstance(parallel, int):
            executer = tuple(
                self.REGISTRY.build(default_args) for _ in range(parallel)
            )
        elif isinstance(parallel, Iterable):
            executer = tuple(
                self.REGISTRY.build(kwargs, default_args)
                for kwargs in parallel
            )
        else:
            raise TypeError(
                "`parallel` must be a bool, int, or Iterable, "
                f"but got {type(parallel)}"
            )
        self._executer = executer

    def _forward(self, inputs: tuple, kwargs: dict):
        if isinstance(self._parallel, bool):
            if not self._parallel:
                return self._executer(*inputs, **kwargs)
            return tuple(
                self._executer(*parallel_inputs, **kwargs)
                for parallel_inputs in zip(*inputs)
            )
        return tuple(  # yapf: disable
            executer(*parallel_inputs, **kwargs)
            for executer, *parallel_inputs in zip(self._executer, *inputs)
        )

    def forward(self, message: dict, **kwargs) -> dict:
        inputs = tuple(message[field] for field in self._fields)
        if not isinstance(self._parallel, bool):
            input_len = strict_zip_len(inputs)
            if len(self._executer) != input_len:
                raise ValueError(
                    "Lengths of `inputs` and `self._executer` must be equal, "
                    f"but got input_len={input_len} and len(self._executer)="
                    f"{len(self._executer)}"
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


StepCfg = Union[dict, Step]


class Job:
    STEP_TYPE: Type = Step

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
        if isinstance(steps, dict):
            steps = tuple(  # yapf: disable
                self.STEP_TYPE.build(step, default_args=dict(id_=id_))
                for id_, step in steps.items()
            )
        elif isinstance(steps, Iterable):
            steps = cast(Iterable[StepCfg], steps)
            steps = tuple(self.STEP_TYPE.build(step) for step in steps)
        else:
            raise TypeError(
                "`steps` must be a dict or Iterable, "
                f"but got steps={steps}"
            )

        self._steps = cast(Tuple[Step], steps)

    def forward(self, message: dict, inplace: bool = False) -> dict:
        if not inplace:
            message = dict(message)
        updated_message = dict()
        for step in self._steps:
            updates = step.forward(message)
            message.update(updates)
            updated_message.update(updates)
        return updated_message


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
        self._executer = nn.ModuleList(self._executer)


class ModuleJob(Job, nn.ModuleList):
    STEP_TYPE = ModuleStep

    def __init__(
        self,
        steps: Union[Dict[str, StepCfg], Iterable[StepCfg]],
    ) -> None:
        Job.__init__(self, steps)
        self._steps = cast(Tuple[ModuleStep], self._steps)
        nn.ModuleList.__init__(self, self._steps)  # TODO: clean up
