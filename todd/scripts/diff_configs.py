import argparse
import tempfile
import webbrowser
from difflib import Differ, HtmlDiff
from enum import Enum, auto
from typing import Any, Optional, Sequence

from mmcv import Config


class DiffMode(Enum):
    TEXT = auto()
    HTML = auto()

    def diff(self, a: str, b: str) -> str:
        if self == DiffMode.TEXT:
            diff = '\n'.join(Differ().compare(a, b))
        elif self == DiffMode.HTML:
            diff = HtmlDiff().make_file(a, b)
        return diff


class OutputMode(Enum):
    TERMINAL = auto()
    FILE = auto()
    BROWSER = auto()

    def output(self, diff: str, filepath: Optional[str] = None):
        if self == OutputMode.TERMINAL:
            assert filepath is None
            print(diff)
        elif self == OutputMode.FILE:
            assert filepath is not None
            with open(filepath, 'w') as f:
                f.write(diff)
        elif self == OutputMode.BROWSER:
            assert filepath is None
            with tempfile.NamedTemporaryFile(
                suffix='.html',
                delete=False,
            ) as f:
                f.write(diff.encode('utf-8'))
                webbrowser.open('file://' + f.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Configs")
    parser.add_argument('a')
    parser.add_argument('b')
    # TODO: add mutually exclusive parameter group
    parser.add_argument('--out')
    args = parser.parse_args()
    return args


def load_config(filepath: str) -> Sequence[str]:

    def _sort(obj: Any):
        if isinstance(obj, dict):
            return {k: _sort(obj[k]) for k in sorted(obj.keys())}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_sort(e) for e in obj)
        return obj

    try:
        config = Config.fromfile(filepath)
    except Exception as e:
        print(f"Error when loading {filepath}.")
        raise e
    config = Config(
        cfg_dict=_sort(config._cfg_dict.to_dict()),
        filename=filepath,
    )
    return config.dump().split('\n')


def main():
    args = parse_args()

    a = load_config(args.a)
    b = load_config(args.b)

    if args.out is None:
        diff_mode = DiffMode.TEXT
        output_mode = OutputMode.TERMINAL
        filepath = None
    elif args.out.endswith('.txt'):
        diff_mode = DiffMode.TEXT
        output_mode = OutputMode.FILE
        filepath = args.out
    elif args.out.endswith('.html'):
        diff_mode = DiffMode.HTML
        output_mode = OutputMode.FILE
        filepath = args.out
    elif args.out == 'browser':
        diff_mode = DiffMode.HTML
        output_mode = OutputMode.BROWSER
        filepath = None
    else:
        raise ValueError(f"Unknown output mode: {args.out}.")

    diff = diff_mode.diff(a, b)
    output_mode.output(diff, filepath)


if __name__ == "__main__":
    main()
