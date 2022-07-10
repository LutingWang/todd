from typing import Dict, Optional

import torch.nn as nn

from ..base import WorkflowConfig
from .base import BaseDistiller, DecoratorMixin
from .builder import DISTILLERS


@DISTILLERS.register_module()
class SelfDistiller(DecoratorMixin, BaseDistiller):

    def __init__(
        self,
        student: nn.Module,
        student_hooks: WorkflowConfig = None,
        weight_transfer: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        assert 'hooks' not in kwargs
        assert 'trackings' not in kwargs

        hooks = {}
        if student_hooks is not None:
            hooks[0] = student_hooks

        if weight_transfer is not None:
            weight_transfer = {  # yapf: disable
                'models.0.' + k: 'models.0.' + v
                for k, v in weight_transfer.items()
                if k != '' and v != ''
            }

        super().__init__(
            [student],
            hooks=hooks,
            weight_transfer=weight_transfer,
            **kwargs,
        )
