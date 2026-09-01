__all__ = [
    'get_bytes',
]

from io import BytesIO

import requests


def get_bytes(url: str) -> BytesIO:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return BytesIO(response.content)
