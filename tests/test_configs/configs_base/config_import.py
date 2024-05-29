from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.distributed import fsdp
    _import_ = None  # pylint: disable=invalid-name
else:
    torch = _import_('torch')
    fsdp = _import_('torch.distributed.fsdp')
