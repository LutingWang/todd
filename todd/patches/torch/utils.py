__all__ = [
    'set_epoch',
]

from torch.utils.data import DataLoader


def set_epoch(dataloader: DataLoader, epoch: int) -> None:
    samplers = [
        dataloader.sampler,
        dataloader.batch_sampler,
        getattr(dataloader.batch_sampler, 'sampler', None),
    ]
    for sampler in samplers:
        set_epoch_ = getattr(sampler, 'set_epoch', None)
        if set_epoch_ is not None:
            set_epoch_(epoch)
