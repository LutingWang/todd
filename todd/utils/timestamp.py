__all__ = ['get_timestamp']

from datetime import datetime

import torch.distributed

from .torch import get_rank, get_world_size


def get_timestamp() -> str:
    if get_rank() == 0:
        timestamp = datetime.now().astimezone().isoformat()
        timestamp = timestamp.replace(':', '-').replace('+', '-')
        timestamp = timestamp.replace('.', '_')
        if get_world_size() <= 1:
            return timestamp
    else:
        timestamp = None
    object_list = [timestamp]
    torch.distributed.broadcast_object_list(object_list, 0)
    return object_list[0]
