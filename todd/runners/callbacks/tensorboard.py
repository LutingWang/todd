from collections import defaultdict
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from ...base import CallbackRegistry, Config
from ...utils import get_rank
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class TensorBoardCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        summary_writer: Config,
        main_tag: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._summary_writer_config = summary_writer
        self._main_tag = main_tag

    def init(self) -> None:
        super().init()
        if get_rank() > 0:
            return
        log_dir = self._runner.work_dir / 'tensorboard'
        self._summary_writer = SummaryWriter(
            log_dir,
            **self._summary_writer_config,
        )

    def _tag(self, tag: str) -> str:
        assert tag
        return self._main_tag + '/' + tag

    def before_run_iter(self, batch, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if get_rank() == 0 and self._should_run_iter():
            memo['tensorboard'] = defaultdict(list)

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if 'tensorboard' not in memo:
            return
        tensorboard: defaultdict[str, list[dict[str, Any]]] = \
            memo.pop('tensorboard')
        for entry_type, entries in tensorboard.items():
            add = getattr(self._summary_writer, 'add_' + entry_type)
            for entry in entries:
                if 'tag' in entry:
                    entry['tag'] = self._tag(entry['tag'])
                if 'main_tag' in entry:
                    entry['main_tag'] = self._tag(entry['main_tag'])
                if 'tags' in entry:
                    entry['tags'] = list(map(self._tag, entry['tags']))
                add(**entry)
