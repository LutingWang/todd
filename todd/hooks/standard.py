from .base import HOOKS, BaseHook


@HOOKS.register_module()
class StandardHook(BaseHook):

    def _reset(self):
        self.__tensor = None

    def _tensor(self):
        return self.__tensor

    def _register_tensor(self, tensor) -> None:
        self.__tensor = tensor
