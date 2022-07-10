from typing import Dict, List, Optional

import torch.nn as nn
from mmcv.runner import ModuleList

from ..base import WorkflowConfig
from ..reproduction import freeze_model
from .base import BaseDistiller, DecoratorMixin
from .builder import DISTILLERS


@DISTILLERS.register_module()
class MultiTeacherDistiller(DecoratorMixin, BaseDistiller):

    def __init__(
        self,
        student: nn.Module,
        online_teachers: List[nn.Module] = None,
        offline_teachers: List[nn.Module] = None,
        student_hooks: WorkflowConfig = None,
        online_teacher_hooks: Optional[Dict[int, WorkflowConfig]] = None,
        offline_teacher_hooks: Optional[Dict[int, WorkflowConfig]] = None,
        **kwargs,
    ):
        assert 'hooks' not in kwargs

        online_teachers = [] if online_teachers is None else online_teachers
        online_teacher_slice = slice(1, 1 + len(online_teachers))

        offline_teachers = [] if offline_teachers is None else offline_teachers
        for offline_teacher in offline_teachers:
            freeze_model(offline_teacher)
        offline_teacher_slice = slice(online_teacher_slice.stop, None)

        hooks = {}
        if student_hooks is not None:
            hooks[0] = student_hooks
        if online_teacher_hooks is not None:
            hooks.update({  # yapf: disable
                online_teacher_slice.start + i: hook
                for i, hook in online_teacher_hooks.items()
            })
        if offline_teacher_hooks is not None:
            hooks.update({  # yapf: disable
                offline_teacher_slice.start + i: hook
                for i, hook in offline_teacher_hooks.items()
            })

        super().__init__(
            [student] + online_teachers + offline_teachers,
            hooks=hooks,
            **kwargs,
        )
        self._online_teachers = ModuleList(online_teachers)
        self._online_teacher_slice = online_teacher_slice
        self._offline_teacher_slice = offline_teacher_slice

    @property
    def student(self) -> nn.Module:
        return self.models[0]

    @property
    def teachers(self) -> List[nn.Module]:
        return self.models[1:]

    @property
    def online_teachers(self) -> List[nn.Module]:
        return self.models[self._online_teacher_slice]

    @property
    def offline_teachers(self) -> List[nn.Module]:
        return self.models[self._offline_teacher_slice]


@DISTILLERS.register_module()
class SingleTeacherDistiller(MultiTeacherDistiller):

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        teacher_hooks: WorkflowConfig = None,
        teacher_online: bool = False,
        **kwargs,
    ):
        assert 'online_teacher_hooks' not in kwargs
        assert 'offline_teacher_hooks' not in kwargs
        assert 'online_teacher_weight_transfer' not in kwargs

        arg_prefix = ['offline', 'online'][teacher_online]
        kwargs[arg_prefix + '_teachers'] = [teacher]
        if teacher_hooks is not None:
            kwargs[arg_prefix + '_teacher_hooks'] = {0: teacher_hooks}
        super().__init__(student, **kwargs)

    @property
    def teacher(self) -> nn.Module:
        return self.teachers[0]
