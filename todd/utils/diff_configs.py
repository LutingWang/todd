from difflib import Differ, HtmlDiff
from typing import Any, Iterator, Sequence

from mmcv import Config


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
    config = Config(cfg_dict=_sort(config._cfg_dict.to_dict()), filename=filepath)
    return config.dump().split('\n')


class ConfigDiffer(Differ):
    def compare(self, a: str, b: str) -> Iterator[str]:
        return super().compare(load_config(a), load_config(b))


class ConfigHtmlDiff(HtmlDiff):
    def make_file(self, a: str, b: str, *args, **kwargs) -> str:
        return super().make_file(load_config(a), load_config(b), *args, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Configs")
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--out', required=False)
    args = parser.parse_args()
    if args.out is not None and args.out.endswith('.html'):
        diff = ConfigHtmlDiff().make_file(args.a, args.b)
    else:
        diff = '\n'.join(ConfigDiffer().compare(args.a, args.b))
    if args.out is None:
        print(diff)
    else:
        with open(args.out, 'w') as f:
            f.write(diff)
