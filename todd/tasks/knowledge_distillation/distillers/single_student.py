__all__ = [
    'SingleStudentDistiller',
    'MultiTeacherDistiller',
    'SingleTeacherDistiller',
    'SelfDistiller',
    'StudentMixin',
]

from abc import ABC
from typing import Generic, Iterable, Mapping, TypeVar

from torch import nn

from todd import Config

from ..registries import KDDistillerRegistry
from .base import BaseDistiller


class SingleStudentDistiller(BaseDistiller, ABC):

    def __init__(
        self,
        *args,
        student: nn.Module,
        teachers: Iterable[nn.Module],
        student_hook: Config,
        teacher_hooks: Iterable[Config],
        **kwargs,
    ) -> None:
        models = (student, ) + tuple(teachers)
        hooks = (student_hook, ) + tuple(teacher_hooks)
        super().__init__(*args, models=models, hook_pipelines=hooks, **kwargs)

    @property
    def student(self) -> nn.Module:
        return self._models[0]


@KDDistillerRegistry.register_()
class MultiTeacherDistiller(SingleStudentDistiller, ABC):

    def __init__(
        self,
        *args,
        online_teachers: Iterable[nn.Module],
        offline_teachers: Iterable[nn.Module],
        online_teacher_hooks: Iterable[Config],
        offline_teacher_hooks: Iterable[Config],
        **kwargs,
    ) -> None:
        online_teachers = tuple(online_teachers)
        offline_teachers = tuple(offline_teachers)
        teachers = online_teachers + offline_teachers
        online_teacher_hooks = tuple(online_teacher_hooks)
        offline_teacher_hooks = tuple(offline_teacher_hooks)
        teacher_hooks = online_teacher_hooks + offline_teacher_hooks
        super().__init__(
            *args,
            teachers=teachers,
            teacher_hooks=teacher_hooks,
            **kwargs,
        )
        self._num_online_teachers = len(online_teachers)

        for offline_teacher in offline_teachers:
            offline_teacher.requires_grad_(False)
            offline_teacher.eval()
        self.add_module('_teachers', nn.ModuleList(online_teachers))

    @property
    def teachers(self) -> tuple[nn.Module, ...]:
        return self.models[1:]

    @property
    def online_teachers(self) -> tuple[nn.Module, ...]:
        return self.models[1:1 + self._num_online_teachers]

    @property
    def offline_teachers(self) -> tuple[nn.Module, ...]:
        return self.models[1 + self._num_online_teachers:]


@KDDistillerRegistry.register_()
class SingleTeacherDistiller(SingleStudentDistiller, ABC):

    def __init__(
        self,
        *args,
        teacher: nn.Module,
        teacher_hook: Config,
        online: bool = False,
        **kwargs,
    ) -> None:
        teachers = (teacher, )
        teacher_hooks = (teacher_hook, )
        super().__init__(
            *args,
            teachers=teachers,
            teacher_hooks=teacher_hooks,
            **kwargs,
        )
        if online:
            self.add_module('_teacher', teacher)
        else:
            teacher.requires_grad_(False)
            teacher.eval()

    @property
    def teacher(self) -> nn.Module:
        return self.models[1]


@KDDistillerRegistry.register_()
class SelfDistiller(SingleStudentDistiller, ABC):

    def __init__(
        self,
        *args,
        weight_transfer: Mapping[str, str] | None = None,
        **kwargs,
    ) -> None:
        if weight_transfer is not None:
            weight_transfer = {
                '.student' + k: '.student' + v
                for k, v in weight_transfer.items()
            }
        super().__init__(
            *args,
            teachers=tuple(),
            teacher_hooks=tuple(),
            weight_transfer=weight_transfer,
            **kwargs,
        )


T = TypeVar('T', bound=SingleStudentDistiller)


class StudentMixin(Generic[T]):

    def __init__(self, *args, distiller: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._build_distiller(distiller)

    def _build_distiller(self, config: Config) -> None:
        self._distiller: T = KDDistillerRegistry.build(config, student=self)

    @property
    def distiller(self) -> T:
        return self._distiller

    @property
    def sync_apply(self) -> bool:
        return False
