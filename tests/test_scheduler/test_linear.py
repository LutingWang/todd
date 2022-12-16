import pytest

from todd import Config, Store
from todd.schedulers import BaseScheduler, SchedulerRegistry


class TestLinearScheduler:
    pass


class TestConstantScheduler:
    pass


class TestWarmupScheduler:

    @pytest.fixture(scope='class')
    def scheduler(self) -> BaseScheduler:
        scheduler = SchedulerRegistry.build(
            Config(
                type='WarmupScheduler',
                value=1,
                iter_=10,
            )
        )
        return scheduler

    @pytest.mark.usefixtures('setup_teardown_iter')
    def test_value(self, scheduler: BaseScheduler) -> None:
        Store.ITER = 1
        assert scheduler.value == 0.1
        Store.ITER = 11
        assert scheduler.value == 1


class TestEarlyStopScheduler:
    pass


class TestDecayScheduler:
    pass
