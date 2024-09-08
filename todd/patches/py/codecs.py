__all__ = [
    'encode_filename',
    'decode_filename',
]

import codecs
import urllib.parse

ASCII = 'ascii'
FILENAME = 'filename'


def filename_codec(name: str) -> codecs.CodecInfo | None:
    if name != FILENAME:
        return None

    def encode(s: str, *args, **kwargs) -> tuple[bytes, int]:
        f = urllib.parse.quote(s, safe='').encode(ASCII)
        return f, len(s)

    def decode(f: bytes, *args, **kwargs) -> tuple[str, int]:
        s = urllib.parse.unquote(f.decode(ASCII))
        return s, len(f)

    return codecs.CodecInfo(encode, decode, name=FILENAME)


codecs.register(filename_codec)


def encode_filename(s: str) -> str:
    return codecs.encode(s, FILENAME).decode(ASCII)


def decode_filename(f: str) -> str:
    return codecs.decode(f.encode(ASCII), FILENAME)
