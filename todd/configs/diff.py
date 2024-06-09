__all__: list[str] = []

import argparse
import tempfile
import webbrowser
from typing import cast

from ..registries import ConfigRegistry
from .serialize import SerializeMixin


def diff_cli() -> None:
    parser = argparse.ArgumentParser(description="Compare Configs")
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--type', default='py')
    parser.add_argument('--out', default='terminal')
    args = parser.parse_args()

    config_type = cast(type[SerializeMixin], ConfigRegistry[args.type])
    a = config_type.load(args.a)
    b = config_type.load(args.b)
    out: str = args.out

    diff = a.diff(b, out == 'browser' or out.endswith('.html'))
    if out == 'terminal':
        print(diff)
    elif out == 'browser':
        with tempfile.NamedTemporaryFile(
            suffix='.html',
            delete=False,
        ) as html_file:
            html_file.write(diff.encode('utf-8'))
            webbrowser.open('file://' + html_file.name)
    else:
        with open(out, 'w') as f:
            f.write(diff)
