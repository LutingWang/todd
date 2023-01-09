from todd import Config
from todd.losses.schedulers import SchedulerRegistry, WarmupScheduler


class TestLinearScheduler:
    pass


class TestConstantScheduler:
    pass


class TestWarmupScheduler:

    def test_value(self) -> None:
        scheduler: WarmupScheduler = SchedulerRegistry.build(
            Config(type='WarmupScheduler', end=10),
        )
        scheduler.steps = 1
        assert scheduler() == 0.1
        scheduler.steps = 11
        assert scheduler() == 1


class TestEarlyStopScheduler:
    pass


class TestDecayScheduler:
    pass
