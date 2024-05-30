__all__ = [
    'CheckCallback',
]

from ...patches.torch import get_rank
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback

# TODO: check if the model has grad after each iteration


@CallbackRegistry.register_()
class CheckCallback(BaseCallback):

    def before_run(self, memo: Memo) -> None:
        if get_rank() == 0:
            requires_grad_parameters = [
                repr(name)
                for name, parameter in
                self.trainer.strategy.module.named_parameters()
                if parameter.requires_grad
            ]
            self.trainer.logger.debug(
                'Requires grad parameters\n'
                + ', '.join(requires_grad_parameters),
            )

            training_modules = [
                repr(name) for name, module in
                self.trainer.strategy.module.named_modules() if module.training
            ]
            self.trainer.logger.debug(
                'Training modules\n' + ', '.join(training_modules),
            )

        super().before_run(memo)
