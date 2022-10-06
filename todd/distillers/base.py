__all__ = [
    'DISTILLERS',
    'BaseDistiller',
    'DistillableProto',
    'build_metaclass',
]

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import torch
import torch.nn as nn

from ..base import (
    Message,
    Module,
    ModuleList,
    Registry,
    Workflow,
    WorkflowConfig,
    transfer_weights,
)
from ..base.workflows import WorkflowSpec
from ..hooks import BaseHook

T = TypeVar('T', bound='BaseDistiller')


class BaseDistiller(Module):
    DEBUG_PREFIX = '_debug_'

    @staticmethod
    def workflow_to_module(workflow: Workflow) -> nn.Module:
        modules = ModuleList()
        for job in workflow:
            step_manager = job.step_manager
            if hasattr(step_manager, 'step'):
                module = step_manager.step  # type: ignore[attr-defined]
            elif hasattr(step_manager, 'steps'):
                module = ModuleList(
                    step_manager.steps,  # type: ignore[attr-defined]
                )
            else:
                raise TypeError(
                    f'{step_manager} has no attribute `step` or `steps`.'
                )
            modules.append(module)
        return modules

    def __init__(
        self,
        models: List[nn.Module],
        *,
        hooks: Dict[int, WorkflowConfig],
        adapts: WorkflowConfig,
        losses: WorkflowConfig,
        weight_transfer: Optional[Dict[str, str]] = None,
    ):
        Module.__init__(self)
        self._models = models

        self._hookflows: Dict[int, Workflow] = {  # yapf: disable
            i: Workflow.build('hooks', hooks_config)
            for i, hooks_config in hooks.items()
        }
        self._adaptflow = Workflow.build('adapts', adapts)
        self._lossflow = Workflow.build('losses', losses)

        if weight_transfer is not None:
            transfer_weights(self, weight_transfer)

        for i, workflow in self._hookflows.items():
            for job in workflow:
                for step in job:
                    cast(BaseHook, step).bind(self._models[i])

        self.add_module(
            '_adapts',
            self.workflow_to_module(self._adaptflow),
        )
        self.add_module(
            '_losses',
            self.workflow_to_module(self._lossflow),
        )

    @property
    def models(self) -> List[nn.Module]:
        return self._models

    def _apply(self: T, fn: Callable[..., None]) -> T:
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn)
        return super()._apply(fn)

    def hookflows(self) -> Iterator[Workflow]:
        return iter(self._hookflows.values())

    def hooks(self) -> Generator[BaseHook, None, None]:
        for workflow in self.hookflows():
            for job in workflow:
                yield from job

    def track_tensors(self) -> None:
        for hook in filter(
            lambda hook: hook.tracking_mode,
            self.hooks(),
        ):
            hook.track_tensor()

    def tensors(self) -> Message:
        tensors: Message = dict()
        for job in self.hookflows():
            job(tensors)
        return tensors

    def reset(self) -> None:
        for hook in self.hooks():
            hook.reset()

    def spec(self) -> WorkflowSpec:
        inputs: Set[str] = set()
        outputs: Set[str] = set()
        for i, hookflow in self._hookflows.items():
            hookflow_spec = hookflow.spec()
            assert len(hookflow_spec.inputs) == 0
            assert outputs.isdisjoint(hookflow_spec.outputs)
            outputs |= hookflow_spec.outputs
        adaptflow_spec = self._adaptflow.spec()
        inputs |= adaptflow_spec.inputs - outputs
        outputs |= adaptflow_spec.outputs
        lossflow_spec = self._lossflow.spec()
        inputs |= lossflow_spec.inputs - outputs
        outputs |= lossflow_spec.outputs
        return WorkflowSpec(inputs, outputs)

    def distill(
        self,
        custom_tensors: Optional[Dict[str, torch.Tensor]] = None,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if debug:
            expected_inputs = self.spec().inputs
            custom_inputs = (
                set()
                if custom_tensors is None else set(custom_tensors.keys())
            )
            if len(expected_inputs ^ custom_inputs):
                warnings.warn(
                    f"Missing inputs {expected_inputs - custom_inputs}\n"
                    f"Unexpected inputs {custom_inputs - expected_inputs}\n"
                )

        tensors = self.tensors()
        if custom_tensors is not None:
            tensors.update(custom_tensors)
        self._adaptflow(tensors)
        losses = self._lossflow(tensors.copy())

        if debug:
            losses.update({
                self.DEBUG_PREFIX + k: v
                for k, v in tensors.items()
            })
        return losses


DISTILLERS: Registry[BaseDistiller] = Registry(
    'distillers',
    base=BaseDistiller,
)


class DistillableProto(Protocol):
    _distiller: BaseDistiller

    @property
    def distiller(self) -> BaseDistiller:
        ...

    @property
    def sync_apply(self) -> bool:
        ...


def build_metaclass(
    distiller_cls: Type[BaseDistiller],
    supermetaclass: Type[type] = type,
) -> type:

    class MetaClass(supermetaclass):  # type: ignore[valid-type, misc]
        NAMESPACE = dict(
            distiller=property(
                lambda x: cast(DistillableProto, x)._distiller,
            ),
            sync_apply=property(lambda _: False),
        )

        def __new__(
            meta_cls,
            cls: str,
            bases: Tuple[type, ...],
            namespace: Dict[str, Any],
            **kwargs: Any,
        ) -> 'MetaClass':
            assert len(namespace.keys() & meta_cls.NAMESPACE.keys()) == 0
            namespace.update(meta_cls.NAMESPACE)
            return super().__new__(
                meta_cls,
                cls,
                bases,
                namespace,
                **kwargs,
            )

        def __call__(
            cls,
            *args,
            **kwargs,
        ) -> DistillableProto:
            if 'distiller' not in kwargs:
                raise RuntimeError('`distiller` is required')
            distiller: Dict[str, Any] = kwargs.pop('distiller')
            obj = super().__call__(*args, **kwargs)
            obj = cast(DistillableProto, obj)
            obj._distiller = distiller_cls(  # type: ignore[call-arg]
                student=obj,
                **distiller,
            )
            return obj

    return MetaClass
