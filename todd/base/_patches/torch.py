import torch

from .._extensions import get_logger

if torch.__version__ < '1.7.0':
    get_logger().warning(
        "Monkey patching `torch.maximum` and `torch.minimum`.",
    )
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min
