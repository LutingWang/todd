__all__ = [
    'HTML4',
]

import enum


# https://www.w3.org/TR/html401/types.html#h-6.5
class HTML4(enum.StrEnum):
    BLACK = '#000000'
    SILVER = '#C0C0C0'
    GRAY = '#808080'
    WHITE = '#FFFFFF'
    MAROON = '#800000'
    RED = '#FF0000'
    PURPLE = '#800080'
    FUCHSIA = '#FF00FF'
    GREEN = '#008000'
    LIME = '#00FF00'
    OLIVE = '#808000'
    YELLOW = '#FFFF00'
    NAVY = '#000080'
    BLUE = '#0000FF'
    TEAL = '#008080'
    AQUA = '#00FFFF'
