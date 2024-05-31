__all__ = [
    'PALETTE',
]

from .constants import HTML4
from .rgba import RGB

PALETTE = [RGB.from_(color.value) for color in HTML4]
