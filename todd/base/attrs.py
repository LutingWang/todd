import contextlib
from typing import Any, Generator

__all__ = [
    'getattr_recur',
    'setattr_recur',
    'setattr_temp',
]


def hasattr_recur(obj: Any, attr: str, allow_list: bool = False) -> bool:
    if attr == '':
        return True
    for a in attr.split('.'):
        if allow_list and a.isnumeric():
            if int(a) not in obj:
                return False
        else:
            if not hasattr(obj, a):
                return False
    return True


def getattr_recur(obj: Any, attr: str, allow_list: bool = False) -> Any:
    if attr == '':
        return obj
    if not allow_list:
        return eval('obj.' + attr)
    for a in attr.split('.'):
        obj = obj[int(a)] if a.isnumeric() else eval('obj.' + a)
    return obj


def setattr_recur(
    obj: Any,
    attr: str,
    value: Any,
    allow_list: bool = False,
) -> None:
    if '.' in attr:
        attr, tmp = attr.rsplit('.', 1)
        obj = getattr_recur(obj, attr, allow_list)
        attr = tmp
    if allow_list and attr.isnumeric:
        obj[int(attr)] = value
    else:
        setattr(obj, attr, value)


def delattr_recur(obj: Any, attr: str, allow_list: bool = False) -> None:
    if '.' in attr:
        attr, tmp = attr.rsplit('.', 1)
        obj = getattr_recur(obj, attr, allow_list)
        attr = tmp
    if allow_list and attr.isnumeric:
        del obj[int(attr)]
    else:
        delattr(obj, attr)


@contextlib.contextmanager
def setattr_temp(
    obj: Any,
    attr: str,
    value: Any,
    allow_list: bool = False,
) -> Generator[None, None, None]:
    if hasattr_recur(obj, attr, allow_list):
        prev = getattr_recur(obj, attr, allow_list)
        setattr_recur(obj, attr, value, allow_list)
        yield
        setattr_recur(obj, attr, prev, allow_list)
    else:
        setattr_recur(obj, attr, value, allow_list)
        yield
        delattr_recur(obj, attr, allow_list)
