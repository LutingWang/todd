from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.distributed.fsdp as fsdp
    _import_ = None
else:
    torch = _import_('torch')
    fsdp = _import_('torch.distributed.fsdp')
