__all__ = [
    'PretrainedMixin',
]

from abc import ABC

from torch import nn

from ...patches.torch import load_state_dict
from ...utils import StateDictConverter


class PretrainedMixin(nn.Module, ABC):
    STATE_DICT_CONVERTER: type[StateDictConverter]

    def load_pretrained(self, *args, **kwargs) -> None:
        converter = self.STATE_DICT_CONVERTER(module=self)
        state_dict = converter.load(*args, **kwargs)
        state_dict = converter.convert(state_dict)
        load_state_dict(self, state_dict, strict=False)
