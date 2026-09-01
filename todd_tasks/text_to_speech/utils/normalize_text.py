__all__ = [
    'normalize_text',
]

import pathlib

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

cache_dir = pathlib.Path(__file__).with_suffix('')
cache_dir.mkdir(parents=True, exist_ok=True)

ZH_NORMALIZER = ZhNormalizer(
    cache_dir=cache_dir,
    remove_erhua=False,
    full_to_half=False,
)
EN_NORMALIZER = EnNormalizer()


def normalize_text(text: str) -> str:
    if text.isascii():
        text = EN_NORMALIZER.normalize(text)
        return text
    text = ZH_NORMALIZER.normalize(text)
    return text
