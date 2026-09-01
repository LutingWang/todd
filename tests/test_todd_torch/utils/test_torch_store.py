from typing import Any

from todd.utils import Store as BaseStore
from todd.utils import StoreMeta
from todd_torch.utils.stores import Store


def test_store(monkeypatch: Any) -> None:
    assert isinstance(Store, StoreMeta)
    assert not issubclass(Store, BaseStore)
    assert not hasattr(Store, 'DRY_RUN')
    assert isinstance(Store.DEVICE, str)

    monkeypatch.setenv('DRY_RUN', 'True')
    assert BaseStore.DRY_RUN is True
    assert not hasattr(Store, 'DRY_RUN')
