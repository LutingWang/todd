__all__ = [
    'normalize_text',
]

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

ZH_NORMALIZER = ZhNormalizer(
    overwrite_cache=True,
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
