from typing import Iterable, Sized

__all__ = [
    'strict_zip_len',
]


def strict_zip_len(iterable: Iterable[Sized]) -> int:
    """Length check for zip.

    Before Python 3.10, zip() stops when the shortest element is exhausted.
    This function checks the length of all elements and raises an error if
    they are not the same.

    Args:
        iterable: Iterable of sized elements.

    Returns:
        Length of the sized elements.

    Raises:
        ValueError: If the length of the sized elements are not the same.
    """
    lens = {len(e) for e in iterable}
    if len(lens) > 1:
        raise ValueError(f'Lengths of iterables are not the same: {lens}')
    return lens.pop()
