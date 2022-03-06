from typing import Dict, Optional
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
        weight_transfer: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        assert 'hooks' not in kwargs 
        assert 'trackings' not in kwargs

        hooks = {}
        if student_hooks is not None:
            hooks[0] = student_hooks

        trackings = {}
        if student_trackings is not None:
            trackings[0] = student_trackings

        if weight_transfer is not None:
            weight_transfer = {
                'models.0.' + k: 'models.0.' + v 
                for k, v in weight_transfer.items()
                if k is not '' and v is not ''
            }

        super().__init__(
            [student], 
            hooks=hooks, 
            trackings=trackings, 
            weight_transfer=weight_transfer, 
            **kwargs,
        )
