from typing import Iterable, Optional, Union

import torch.nn as nn

from ..hooks import HookModule, TrackingModule

from .base import InterfaceDistiller
from .builder import DISTILLERS


@DISTILLERS.register_module()
class SelfDistiller(InterfaceDistiller):
    def __init__(
        self, 
        student: nn.Module, 
        student_hooks: Optional[Union[HookModule, Iterable[Optional[dict]]]] = None, 
        student_trackings: Optional[Union[TrackingModule, Iterable[Optional[dict]]]] = None, 
        **kwargs,
    ):
        assert not kwargs.get('hooks') and not kwargs.get('trackings')

        hooks = {}
        if student_hooks is not None:
            hooks[0] = student_hooks
        kwargs['hooks'] = hooks

        trackings = {}
        if student_trackings is not None:
            trackings[0] = student_trackings
        kwargs['trackings'] = trackings

        super().__init__([student], **kwargs)
