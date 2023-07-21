__all__ = ['StateDictMixin']

from typing import Any, Mapping


class StateDictMixin:

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        pass
