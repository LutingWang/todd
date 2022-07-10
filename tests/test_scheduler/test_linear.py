import pytest

from todd.schedulers import SCHEDULERS, BaseScheduler


class TestLinearScheduler:
    pass


class TestConstantScheduler:
    pass


class TestWarmupScheduler:

    @pytest.fixture(scope='class')
    def scheduler(self) -> BaseScheduler:
        scheduler = SCHEDULERS.build(
            dict(
                type='WarmupScheduler',
                value=1,
                iter_=10,
            )
        )
        return scheduler

    @pytest.mark.usefixtures('setup_teardown_iter')
    @pytest.mark.parametrize(
        'setup_value,teardown_value',
        [(i, None) for i in range(11)],
    )
    def test_warmup(
        self,
        setup_value: int,
        scheduler: BaseScheduler,
    ) -> None:
        assert scheduler.value == setup_value / 10

    @pytest.mark.usefixtures('setup_teardown_iter')
    @pytest.mark.parametrize(
        'setup_value,teardown_value',
        [(11, None)],
    )
    def test_after(self, setup_value: int, scheduler: BaseScheduler) -> None:
        assert scheduler.value == 1


class TestEarlyStopScheduler:
    pass


class TestDecayScheduler:
    pass
