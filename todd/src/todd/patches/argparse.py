__all__ = [
    'DictAction',
]

import argparse
from typing import Any, Sequence, cast


class DictAction(argparse.Action):
    """``argparse`` action to parse arguments in the form of key-value pairs.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--dict', action=DictAction)
        DictAction(...)
        >>> parser.parse_args('--dict key1::value1 key2::value2'.split())
        Namespace(dict={'key1': 'value1', 'key2': 'value2'})
    """

    def __init__(self, *args, **kwargs) -> None:
        assert 'nargs' not in kwargs
        kwargs['nargs'] = argparse.ZERO_OR_MORE
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values=None,
        option_string: str | None = None,
    ) -> None:
        values = cast(Sequence[str], values)
        value_dict: dict[str, Any] = dict()
        for value in values:
            k, v = value.split(':', 1)
            k = k.strip()
            v = v[1:] if v.startswith(':') else eval(v)  # nosec B307
            value_dict[k] = v
        setattr(namespace, self.dest, value_dict)
