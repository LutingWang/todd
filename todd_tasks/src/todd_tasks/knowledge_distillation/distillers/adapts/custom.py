__all__ = [
    'Custom',
]

import string

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


@KDAdaptRegistry.register_()
class Custom(BaseAdapt):
    """Custom adaptation described using patterns.

    Examples:
        >>> adapt = Custom(pattern='a + b * c')
        >>> adapt(1, 2, 3)
        7
        >>> adapt = Custom(pattern='a + var1 * var2')
        >>> adapt(1, var1=2, var2=3)
        7
        >>> adapt = Custom(pattern='a + b * c')
        >>> adapt(1, 2, 3, b=4)
        Traceback (most recent call last):
            ...
        RuntimeError: {'b'}
    """

    def __init__(self, *args, pattern: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pattern = pattern

    def forward(self, *args, **kwargs):
        if keys := set(string.ascii_letters) & kwargs.keys():
            raise RuntimeError(keys)
        kwargs |= zip(string.ascii_letters, args)
        return eval(self._pattern, None, kwargs)  # nosec B307
