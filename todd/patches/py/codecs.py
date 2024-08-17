__all__ = [
    'encode_filename',
    'decode_filename',
]

import codecs
import urllib.parse


def filename_codec(name: str) -> codecs.CodecInfo | None:
    if name != 'filename':
        return None

    def encode(s: str, *args, **kwargs) -> tuple[bytes, int]:
        f = urllib.parse.quote(s, safe='').encode('ascii')
        return f, len(s)

    def decode(f: bytes, *args, **kwargs) -> tuple[str, int]:
        s = urllib.parse.unquote(f.decode('ascii'))
        return s, len(f)

    return codecs.CodecInfo(encode, decode, name=name)


codecs.register(filename_codec)


def encode_filename(s: str) -> str:
    return codecs.encode(s, 'filename').decode('ascii')


def decode_filename(f: str) -> str:
    return codecs.decode(f.encode('ascii'), 'filename')
