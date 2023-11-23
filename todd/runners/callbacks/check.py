__all__ = [
    'CheckCallback',
]

from ...base import CallbackRegistry
from ..types import Memo
from .base import BaseCallback

# TODO: check if the model has grad


@CallbackRegistry.register_()
class CheckCallback(BaseCallback):

    def before_run(self, memo: Memo) -> None:
        requires_grad_parameters = [
            name for name, parameter in
            self.trainer.strategy.module.named_parameters()
            if parameter.requires_grad
        ]
        self.trainer.logger.info(
            'Requires grad parameters:\n'
            + ', '.join(requires_grad_parameters),
        )

        training_modules = [
            name
            for name, module in self.trainer.strategy.module.named_modules()
            if module.training
        ]
        self.trainer.logger.info(
            'Training modules:\n' + ', '.join(training_modules),
        )

        super().before_run(memo)
