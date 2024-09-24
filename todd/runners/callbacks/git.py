__all__ = [
    'GitCallback',
]

import subprocess  # nosec B404
from typing import TypeVar

from torch import nn

from ...patches.py_ import run
from ...patches.torch import get_rank
from ...utils import get_timestamp
from ..registries import CallbackRegistry
from .base import BaseCallback

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class GitCallback(BaseCallback[T]):

    def __init__(
        self,
        *args,
        diff: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._diff = diff

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)

        if get_rank() > 0:
            return
        if self._diff is not None:
            args_ = 'git diff'
            if self._diff:
                args_ += f' {self._diff}'
            try:
                diff = run(args_)
            except subprocess.CalledProcessError as e:
                diff = str(e)
                self.runner.logger.error(e)
            else:
                file = (
                    self.runner.work_dir / f'git_diff_{get_timestamp()}.log'
                )
                self.runner.logger.info('Saving git diff to %s', file)
                file.write_text(diff)
