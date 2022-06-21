import string

from .base import ADAPTS, BaseAdapt


@ADAPTS.register_module()
class Custom(BaseAdapt):

    def __init__(self, *args, pattern: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._pattern = pattern

    def forward(self, *args, **kwargs):
        for i, (name, _) in enumerate(zip(string.ascii_letters, args)):
            exec(f'{name} = args[{i}]')
        for key in kwargs:
            if key in locals():
                raise SyntaxError(key)
            exec(f'{key} = kwargs["{key}"]')
        return eval(self._pattern)
