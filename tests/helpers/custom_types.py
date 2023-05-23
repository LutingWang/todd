from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    CustomObject = Any
else:

    class CustomObject:

        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)


class CustomModule(nn.Module):

    def __init__(self, **kwargs: nn.Module) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
