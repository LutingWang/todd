from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Union,
    cast,
    final,
    overload,
)

import torch.nn as nn

from ._extensions import get_logger
from .misc import strict_zip_len
from .registries import Registry

__all__ = [
    'STEPS',
    'Step',
    'StepCfg',
    'Job',
    'JobCfg',
    'Workflow',
]

STEPS: Registry[Callable[..., Any]] = Registry('steps')


@final
class Step:

    @overload
    def __init__(
        self,
        job_id: str,
        id_: str,
        fields: Iterable[str],
        parallel: Union[bool, int, Iterable[dict]] = False,
        **default_args,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        job_id: str,
        id_: Iterable[str],
        fields: Iterable[str],
        **default_args,
    ) -> None:
        ...

    def __init__(
        self,
        job_id,
        id_,
        fields,
        parallel=False,
        **default_args,
    ) -> None:
        self._id = id_
        self._fields = tuple(fields)
        self._parallel = parallel
        self._logger = get_logger()

        if isinstance(parallel, bool):
            self._executor = STEPS.descendent(job_id).build(default_args)
        elif isinstance(parallel, int):
            self._executors = tuple(
                STEPS.descendent(job_id).build(default_args)
                for _ in range(parallel)
            )
        elif isinstance(parallel, Iterable):
            self._executors = tuple(
                STEPS.descendent(job_id).build(kwargs, default_args)
                for kwargs in parallel
            )
        else:
            raise TypeError(
                "`parallel` must be a bool, int, or Iterable, "
                f"but got {type(parallel)}"
            )

    @property
    def parallel(self) -> bool:
        return isinstance(self._parallel, bool) and self._parallel

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

    def to_module(self) -> nn.Module:
        if isinstance(self._parallel, bool):
            return cast(nn.Module, self._executor)
        return nn.ModuleList(self._executors)


StepId = Union[str, Iterable[str]]
StepCfg = Union[Dict[str, Any], Step]


@final
class Job:

    def _build_steps(
        self,
        steps: Union[Dict[StepId, StepCfg], Iterable[StepCfg]],
    ) -> Tuple[Step, ...]:
        if isinstance(steps, dict):
            steps = cast(Dict[StepId, StepCfg], steps)
            return tuple(  # yapf: disable
                step if isinstance(step, Step) else
                Step(self._id, step_id, **step)
                for step_id, step in steps.items()
            )
        if isinstance(steps, Iterable):
            steps = cast(Iterable[StepCfg], steps)
            return tuple(
                step if isinstance(step, Step) else Step(self._id, **step)
                for step in steps
            )
        raise TypeError(
            "`steps` must be a dict or Iterable, "
            f"but got steps={steps}"
        )

    @overload
    def __init__(
        self,
        id_: str,
        steps: Dict[StepId, StepCfg],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        id_: str,
        steps: Iterable[StepCfg],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        id_: str,
        /,
        **steps: StepCfg,
    ) -> None:
        ...

    def __init__(self, id_, steps=None, /, **kwargs) -> None:
        if steps is not None and len(kwargs) > 0:
            raise ValueError("`steps` and `kwargs` cannot be both specified")

        self._id = id_
        self._steps = self._build_steps(
            cast(Dict[StepId, StepCfg], steps or kwargs),
        )

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    def forward(self, message: dict) -> dict:
        updated_message = dict()
        for step in self._steps:
            updates = step.forward(message)
            message.update(updates)
            updated_message.update(updates)
        return updated_message

    def to_module(self) -> nn.Module:
        return nn.ModuleList([step.to_module() for step in self._steps])


JobCfg = Union[Dict[StepId, StepCfg], Iterable[StepCfg], Job]


class Workflow:

    def _build_jobs(self, jobs: Dict[str, JobCfg]) -> Dict[str, Job]:
        jobs = jobs.copy()
        for job_id, job in jobs.items():
            if not isinstance(job, Job):
                jobs[job_id] = Job(job_id, job)
        return cast(Dict[str, Job], jobs)

    @overload
    def __init__(
        self,
        id_: str,
        jobs: Dict[str, JobCfg],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        id_: str,
        /,
        **jobs: JobCfg,
    ) -> None:
        ...

    def __init__(self, id_, jobs=None, /, **kwargs) -> None:
        if jobs is not None and len(kwargs) > 0:
            raise ValueError("`jobs` and `kwargs` cannot be both specified")

        self._id = id_
        self._jobs = self._build_jobs(jobs or kwargs)

    def has_job(self, job_id: str) -> bool:
        return job_id in self._jobs

    def job(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
