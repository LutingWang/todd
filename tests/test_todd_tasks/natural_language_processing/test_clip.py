import gzip
import inspect
from pathlib import Path

from todd_tasks.natural_language_processing.tokenizers.clip import (
    CLIPTokenizer,
)


def test_bpe_resource() -> None:
    file = Path(inspect.getfile(CLIPTokenizer)).with_name(
        'clip_bpe.txt.gz',
    )
    with gzip.open(file, 'rt') as f:
        assert f.readline().startswith('"bpe_simple_vocab_16e6.txt#version:')
