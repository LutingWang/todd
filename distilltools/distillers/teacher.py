from typing import Dict, Iterable, List, Optional, Union

import torch.nn as nn

from ..hooks import HookModule, TrackingModule

from .base import MixinDistiller
from .builder import DISTILLERS


@DISTILLERS.register_module()
class MultiTeacherDistiller(MixinDistiller):
    def __init__(
        self,
        student: nn.Module,
        online_teachers: List[nn.Module] = None,
        offline_teachers: List[nn.Module] = None,
        student_hooks: Optional[Union[HookModule, Iterable[Optional[dict]]]] = None, 
        student_trackings: Optional[Union[TrackingModule, Iterable[Optional[dict]]]] = None, 
        online_teacher_hooks: Optional[Dict[int, Union[HookModule, Iterable[Optional[dict]]]]] = None, 
        online_teacher_trackings: Optional[Dict[int, Union[TrackingModule, Iterable[Optional[dict]]]]] = None, 
        offline_teacher_hooks: Optional[Dict[int, Union[HookModule, Iterable[Optional[dict]]]]] = None, 
        offline_teacher_trackings: Optional[Dict[int, Union[TrackingModule, Iterable[Optional[dict]]]]] = None, 
        **kwargs,
    ):
        assert not kwargs.get('hooks') and not kwargs.get('trackings')

        online_teachers = [] if online_teachers is None else online_teachers
        online_teacher_slice = slice(1, 1 + len(online_teachers))

        offline_teachers = [] if offline_teachers is None else offline_teachers
        offline_teacher_slice = slice(online_teacher_slice.stop, None)

        hooks = {}
        if student_hooks is not None:
            hooks[0] = student_hooks
        if online_teacher_hooks is not None:
            hooks.update({
                online_teacher_slice.start + i: hook
                for i, hook in online_teacher_hooks.items()
            })
        if offline_teacher_hooks is not None:
            hooks.update({
                offline_teacher_slice.start + i: hook
                for i, hook in offline_teacher_hooks.items()
            })
        kwargs['hooks'] = hooks

        trackings = {}
        if student_trackings is not None:
            trackings[0] = student_trackings
        if online_teacher_trackings is not None:
            trackings.update({
                online_teacher_slice.start + i: tracking
                for i, tracking in online_teacher_trackings.items()
            })
        if offline_teacher_trackings is not None:
            trackings.update({
                offline_teacher_slice.start + i: tracking
                for i, tracking in offline_teacher_trackings.items()
            })
        kwargs['trackings'] = trackings

        super().__init__(
            [student] + online_teachers + offline_teachers, **kwargs,
        )
        self._online_teacher_slice = online_teacher_slice
        self._offline_teacher_slice = offline_teacher_slice

    @property
    def student(self) -> nn.Module:
        return self._models[0]

    @property
    def teachers(self) -> List[nn.Module]:
        return self._models[1:]

    @property
    def online_teachers(self) -> List[nn.Module]:
        return self._models[self._online_teacher_slice]

    @property
    def offline_teachers(self) -> List[nn.Module]:
        return self._models[self._offline_teacher_slice]


@DISTILLERS.register_module()
class SingleTeacherDistiller(MultiTeacherDistiller):
    def __init__(
        self, 
        student: nn.Module, 
        teacher: nn.Module, 
        teacher_hooks: Optional[Union[HookModule, Iterable[Optional[dict]]]] = None,
        teacher_trackings: Optional[Union[TrackingModule, Iterable[Optional[dict]]]] = None,
        teacher_online: bool = False, 
        **kwargs,
    ):
        assert not kwargs.get('online_teacher_hooks') and not kwargs.get('online_teacher_trackings')
        assert not kwargs.get('offline_teacher_hooks') and not kwargs.get('offline_teacher_trackings')

        arg_prefix = ['offline', 'online'][teacher_online]
        kwargs[f'{arg_prefix}_teachers'] = [teacher]
        if teacher_hooks is not None:
            kwargs[f'{arg_prefix}_teacher_hooks'] = {0: teacher_hooks}
        if teacher_trackings is not None:
            kwargs[f'{arg_prefix}_teacher_trackings'] = {0: teacher_trackings}
        super().__init__(student, **kwargs)

    @property
    def teacher(self) -> nn.Module:
        return self.teachers[0]
