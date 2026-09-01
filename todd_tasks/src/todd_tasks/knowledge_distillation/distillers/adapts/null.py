__all__ = [
    'Null',
]

from typing import Any

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


@KDAdaptRegistry.register_()
class Null(BaseAdapt):

    def forward(self, x: Any) -> Any:
        return x
