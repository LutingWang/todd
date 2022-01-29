import functools
from typing import Callable

from typing_extensions import Literal
import torch
from torch.nn._reduction import get_enum


def weight_loss(loss_func: Callable[..., torch.Tensor]):

    @functools.wraps(loss_func)
    def wrapper(
        *args, weight: torch.Tensor = None, 
        reduction: Literal['none', 'mean', 'sum'] = 'mean', **kwargs,
    ):
        loss = loss_func(*args, **kwargs)
        if weight is not None:
            loss = loss * weight
        reduction_enum = get_enum(reduction)  # none: 0, elementwise_mean: 1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()
        raise Exception

    return wrapper
