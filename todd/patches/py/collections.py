__all__ = [
    'AttrDict',
]

from collections import UserDict


class AttrDict(UserDict):

    @classmethod
    def _map(cls, item):
        if isinstance(item, (list, tuple, set)):
            return item.__class__(map(cls._map, item))
        if isinstance(item, dict):
            return cls(item)
        return item

    def __setitem__(self, name: str, value) -> None:
        value = self._map(value)
        super().__setitem__(name, value)

    def __setattr__(self, name: str, value) -> None:
        if name == 'data' or hasattr(self.__class__, name):
            super().__setattr__(name, value)
            return
        self[name] = value

    def __getattr__(self, name: str):
        if name == 'data':  # triggered in `copy.deepcopy`
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e) from e

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(e) from e
