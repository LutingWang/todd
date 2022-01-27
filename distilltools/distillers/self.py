import torch.nn as nn

from ..hooks import HookModuleListCfg

from .base import BaseDistiller, DecoratorMixin
from .builder import DISTILLERS


@DISTILLERS.register_module()
class SelfDistiller(DecoratorMixin, BaseDistiller):
    def __init__(
        self, 
        student: nn.Module, 
        student_hooks: HookModuleListCfg = None, 
        student_trackings: HookModuleListCfg = None, 
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
