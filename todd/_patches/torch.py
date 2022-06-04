import torch

from .logging import get_logger

_logger = get_logger()

if torch.__version__ < '1.7.0':
    _logger.warning("Monkey patching `torch.maximum` and `torch.minimum`.")
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min
