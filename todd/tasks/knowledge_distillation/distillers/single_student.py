__all__ = [
    'SingleStudentDistiller',
    'MultiTeacherDistiller',
    'SingleTeacherDistiller',
    'SelfDistiller',
    'StudentMixin',
    'distiller_decorator',
]

import functools
from abc import ABC
from typing import Generic, Iterable, Mapping, TypeVar

from torch import nn

from todd import Config, RegistryMeta
from todd.bases.registries import Item

from ..registries import KDDistillerRegistry
from ..utils import Pipeline
from .base import BaseDistiller
from .hooks import BaseHook


class SingleStudentDistiller(BaseDistiller, ABC):

    def __init__(
        self,
        *args,
        student: nn.Module,
        teachers: Iterable[nn.Module],
        student_hook_pipeline: Pipeline[BaseHook],
        teachers_hook_pipelines: Pipeline[BaseHook],
        **kwargs,
    ) -> None:
        models = (student, ) + tuple(teachers)
        hook_pipelines = Pipeline(
            processors=(student_hook_pipeline, )
            + teachers_hook_pipelines.processors,
        )
        super().__init__(
            *args,
            models=models,
            hook_pipelines=hook_pipelines,
            **kwargs,
        )

    @property
    def student(self) -> nn.Module:
        return self._models[0]

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        assert 'hook_pipelines' not in config
        config.hook_pipelines = None
        config = super().build_pre_hook(config, registry, item)
        config.pop('hook_pipelines')

        config.teachers_hook_pipelines = cls.build_pipelines(
            config.teachers_hook_pipelines,
        )
        config.student_hook_pipeline = cls.build_pipeline(
            config.student_hook_pipeline,
        )
        return config


@KDDistillerRegistry.register_()
class MultiTeacherDistiller(SingleStudentDistiller, ABC):

    def __init__(
        self,
        *args,
        online_teachers: Iterable[nn.Module],
        offline_teachers: Iterable[nn.Module],
        online_teachers_hook_pipelines: Pipeline[BaseHook],
        offline_teachers_hook_pipelines: Pipeline[BaseHook],
        **kwargs,
    ) -> None:
        online_teachers = tuple(online_teachers)
        offline_teachers = tuple(offline_teachers)
        teachers = online_teachers + offline_teachers
        teachers_hook_pipelines = Pipeline(
            processors=online_teachers_hook_pipelines.processors
            + offline_teachers_hook_pipelines.processors,
        )
        super().__init__(
            *args,
            teachers=teachers,
            teachers_hook_pipelines=teachers_hook_pipelines,
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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        assert 'teachers_hook_pipelines' not in config
        config.teachers_hook_pipelines = None
        config = super().build_pre_hook(config, registry, item)
        config.pop('teachers_hook_pipelines')

        config.online_teachers_hook_pipelines = cls.build_pipelines(
            config.online_teachers_hook_pipelines,
        )
        config.offline_teachers_hook_pipelines = cls.build_pipelines(
            config.offline_teachers_hook_pipelines,
        )
        return config


@KDDistillerRegistry.register_()
class SingleTeacherDistiller(SingleStudentDistiller, ABC):

    def __init__(
        self,
        *args,
        teacher: nn.Module,
        teacher_hook_pipeline: Pipeline[BaseHook],
        online: bool = False,
        **kwargs,
    ) -> None:
        teachers = (teacher, )
        teachers_hook_pipelines = Pipeline(processors=[teacher_hook_pipeline])
        super().__init__(
            *args,
            teachers=teachers,
            teachers_hook_pipelines=teachers_hook_pipelines,
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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        assert 'teachers_hook_pipelines' not in config
        config.teachers_hook_pipelines = None
        config = super().build_pre_hook(config, registry, item)
        config.pop('teachers_hook_pipelines')

        config.teacher_hook_pipeline = cls.build_pipeline(
            config.teacher_hook_pipeline,
        )
        return config


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
            teachers=[],
            teachers_hook_pipelines=Pipeline(processors=[]),
            weight_transfer=weight_transfer,
            **kwargs,
        )

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        assert 'teachers_hook_pipelines' not in config
        config.teachers_hook_pipelines = None
        config = super().build_pre_hook(config, registry, item)
        config.pop('teachers_hook_pipelines')
        return config


T = TypeVar('T', bound=SingleStudentDistiller)


class StudentMixin(Generic[T]):
    _distiller: T

    @property
    def distiller(self) -> T:
        return self._distiller

    @property
    def sync_apply(self) -> bool:
        return False


def distiller_decorator(func):

    @functools.wraps(func)
    def wrapper(
        self: StudentMixin[T],
        *args,
        distiller: Config | None = None,
        **kwargs,
    ) -> None:
        func(self, *args, **kwargs)
        if distiller is not None:
            self._distiller = KDDistillerRegistry.build(
                distiller,
                student=self,
            )

    return wrapper
