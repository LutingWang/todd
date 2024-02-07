__all__ = [
    'GitCallback',
]

import subprocess  # nosec B404

from ...base import CallbackRegistry
from ...utils import get_rank, get_timestamp, subprocess_run
from .base import BaseCallback


@CallbackRegistry.register_()
class GitCallback(BaseCallback):

    def __init__(
        self,
        *args,
        diff: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._diff = diff

    def init(self) -> None:
        super().init()
        if get_rank() > 0:
            return
        if self._diff is not None:
            args = 'git diff'
            if self._diff:
                args += f' {self._diff}'
            try:
                diff = subprocess_run(args)
            except subprocess.CalledProcessError as e:
                diff = str(e)
                self._runner.logger.error(e)
            else:
                file = (
                    self._runner.work_dir / f'git_diff_{get_timestamp()}.log'
                )
                self._runner.logger.info('Saving git diff to %s', file)
                file.write_text(diff)
