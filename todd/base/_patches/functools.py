import functools
import sys

from .._extensions import get_logger

if sys.version_info < (3, 8):
    get_logger().warning("Monkey patching `functools.cached_property`.")

    class cached_property:

        def __init__(self, func):
            self.func = func
            self.__doc__ = func.__doc__

        def __set_name__(self, owner, name):
            self.attrname = name

        def __get__(self, instance, owner=None):
            if self.attrname in instance.__dict__:
                return instance.__dict__[self.attrname]
            val = self.func(instance)
            instance.__dict__[self.attrname] = val
            return val

    functools.cached_property = cached_property

if sys.version_info < (3, 9):
    get_logger().warning("Monkey patching `functools.cache`.")
    functools.cache = functools.lru_cache(maxsize=None)
