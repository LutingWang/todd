__all__ = [
    'Null',
]

from typing import Any

from ..registries import AdaptRegistry
from .base import BaseAdapt


@AdaptRegistry.register_()
class Null(BaseAdapt):

    def forward(self, x: Any) -> Any:
        return x
