__all__ = ['StateDict']

from abc import ABC, abstractmethod
from typing import Any, Mapping


class StateDict(ABC):

    @abstractmethod
    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        pass
