__all__ = [
    'SingleStudentDistiller',
    'MultiTeacherDistiller',
    'SingleTeacherDistiller',
    'SelfDistiller',
    'Student',
]

from typing import Generic, TypeVar, final
from typing_extensions import Self

import torch.nn as nn

from ..base import Config, ModuleList
from ..reproduction import freeze
from .base import BaseDistiller, DistillerRegistry


class SingleStudentDistiller(BaseDistiller):

    @classmethod
    def build(cls, config: Config) -> Self:
        assert 'models' not in config
        assert 'hooks' not in config

        config = config.copy()

        student = config.pop('student'),
        teachers = config.pop('teachers', tuple())
        config.models = student + teachers

        hooks = config.pop('teacher_hooks', dict())
        hooks = {k + 1: v for k, v in hooks.items()}
        if 'student_hooks' in config:
            hooks[0] = config.pop('student_hooks')
        config.hooks = hooks

        return super().build(config)

    @property
    def student(self) -> nn.Module:
        return self.models[0]


@DistillerRegistry.register()
class MultiTeacherDistiller(SingleStudentDistiller):

    @classmethod
    def build(cls, config: Config) -> Self:
        assert 'teachers' not in config
        assert 'teacher_hooks' not in config
        assert 'num_onlines' not in config

        config = config.copy()

        online_teachers = config.pop('online_teachers', tuple())
        offline_teachers = config.pop('offline_teachers', tuple())
        config.teachers = online_teachers + offline_teachers

        config.num_onlines = len(online_teachers)

        teacher_hooks = config.pop('online_hook', dict())
        if 'offline_hooks' in config:
            offline_hooks: Config = config.pop('offline_hooks')
            teacher_hooks.update({
                config.num_onlines + k: v
                for k, v in offline_hooks.items()
            })
        config.teacher_hooks = teacher_hooks

        distiller = super().build(config)

        for offline_teacher in offline_teachers:
            freeze(offline_teacher)
        distiller.add_module('_teachers', ModuleList(online_teachers))

        return distiller

    def __init__(self, *args, num_onlines: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_onlines = num_onlines

    @property
    def teachers(self) -> tuple[nn.Module, ...]:
        return self.models[1:]

    @property
    def online_teachers(self) -> tuple[nn.Module, ...]:
        return self.models[1:1 + self._num_onlines]

    @property
    def offline_teachers(self) -> tuple[nn.Module, ...]:
        return self.models[1 + self._num_onlines:]


@DistillerRegistry.register()
class SingleTeacherDistiller(SingleStudentDistiller):

    @classmethod
    def build(cls, config: Config) -> Self:
        assert 'teachers' not in config

        config = config.copy()

        teacher = config.pop('teacher')
        config.teachers = teacher,

        if 'teacher_hooks' in config:
            config.teacher_hooks = {0: config.teacher_hooks}

        distiller = super().build(config)

        if config.pop('online', False):
            distiller.add_module('_teacher', teacher)
        else:
            freeze(teacher)

        return distiller

    @property
    def teacher(self) -> nn.Module:
        return self.models[1]


@DistillerRegistry.register()
class SelfDistiller(SingleStudentDistiller):

    @classmethod
    def build(cls, config: Config) -> Self:
        assert 'teachers' not in config
        assert 'teacher_hooks' not in config

        config = config.copy()

        if 'weight_transfer' in config:
            weight_transfer: Config = config.weight_transfer
            config.weight_transfer = {
                '.student' + k: '.student' + v
                for k, v in weight_transfer.items()
            }

        return super().build(config)


T = TypeVar('T', bound='SingleStudentDistiller')


class Student(Generic[T]):

    def __init__(self, distiller: Config) -> None:
        self._distiller = DistillerRegistry.build(
            distiller,
            Config(student=self),
        )

    @property
    def distiller(self) -> T:
        return self._distiller

    @property
    @final
    def sync_apply(self) -> bool:
        return False
