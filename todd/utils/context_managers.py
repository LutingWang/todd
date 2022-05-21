from typing import Any, Callable, TypeVar, cast
import functools
import inspect


FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class DecoratorContextManager:
    def __call__(self, wrapped_func: F) -> F:
        if inspect.isgeneratorfunction(wrapped_func):
            raise TypeError(f'@{type(self).__name__}(...) cannot be applied to a generator function.')

        @functools.wraps(wrapped_func)
        def wrapper_func(*args, **kwargs):
            with self:
                return wrapped_func(*args, **kwargs)

        return cast(F, wrapper_func)
