from todd.utils.stores import Store, StoreMeta


def test_store() -> None:
    assert isinstance(Store, StoreMeta)
    assert Store.DRY_RUN is False
    assert not hasattr(Store, 'DEVICE')
