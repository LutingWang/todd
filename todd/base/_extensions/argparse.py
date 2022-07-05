import argparse
from typing import Any, Dict, Optional, Sequence, Union

__all__ = [
    'DictAction',
]


class DictAction(argparse.Action):
    """``argparse`` action to parse arguments in the form of key-value pairs.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--dict', action=DictAction)
        DictAction(...)
        >>> parser.parse_args('--dict key1:"value1" key2:"value2"'.split())
        Namespace(dict={'key1': 'value1', 'key2': 'value2'})
    """

    def __init__(self, *args, nargs=None, **kwargs):
        """
        Args:
            nargs: The number of dictionary arguments that should be consumed.
        """
        if nargs not in [
            None,
            argparse.OPTIONAL,
            argparse.ZERO_OR_MORE,
            argparse.ONE_OR_MORE,
        ]:
            raise ValueError(f"Invalid nargs={nargs}")

        append = nargs in [argparse.ZERO_OR_MORE, argparse.ONE_OR_MORE]
        required = nargs in [None, argparse.ONE_OR_MORE]
        default = [] if append else None
        super().__init__(
            *args,
            nargs=argparse.ZERO_OR_MORE,
            required=required,
            default=default,
            **kwargs,
        )
        self._append = append

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Optional[str] = None,
    ) -> None:
        if not isinstance(values, Sequence):
            raise ValueError(f'values must be a sequence, but got {values}')
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f'values must be strings, but got {values}')
        value_dict: Dict[str, Any] = {}
        for value in values:
            k, v = value.split(':', 1)
            value_dict[k.strip()] = eval(v)
        if self._append:
            value_dict_list = getattr(namespace, self.dest, [])
            value_dict_list.append(value_dict)
            setattr(namespace, self.dest, value_dict_list)
        else:
            setattr(namespace, self.dest, value_dict)
