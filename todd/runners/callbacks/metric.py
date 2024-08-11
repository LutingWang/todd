__all__ = [
    'MetricCallback',
]

from typing import Any, Iterable

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ...patches.torch import get_rank
from ..callbacks import BaseCallback
from ..memo import Memo
from ..metrics import BaseMetric
from ..registries import CallbackRegistry, MetricRegistry


@CallbackRegistry.register_()
class MetricCallback(BuildPreHookMixin, BaseCallback):

    def __init__(
        self,
        *args,
        metrics: Iterable[BaseMetric],  # no need to use ModuleList
        map_model_config: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._metrics = list(metrics)

        if map_model_config is None:
            map_model_config = Config()
        self._map_model_config = map_model_config

    @classmethod
    def metrics_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        metrics = config.metrics
        if isinstance(metrics, Config):
            metrics = [
                MetricRegistry.build_or_return(v, name=k)
                for k, v in metrics.items()
            ]
        else:
            metrics = [
                MetricRegistry.build_or_return(metric) for metric in metrics
            ]
        config.metrics = [m for m in metrics if m is not None]
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.metrics_build_pre_hook(config, registry, item)
        return config

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        for metric in self._metrics:
            metric.bind(*args, **kwargs)
            self.runner.strategy.map_model(metric, self._map_model_config)

    def after_run_iter(self, batch: Any, memo: Memo) -> None:
        for metric in self._metrics:
            memo = metric(batch, memo)
        super().after_run_iter(batch, memo)

    def after_run(self, memo: Memo) -> None:
        metrics = {
            metric.name: metric.summary(memo)
            for metric in self._metrics
        }
        if get_rank() == 0:
            self.runner.logger.info(metrics)
        memo['metrics'] = metrics
        super().after_run(memo)
