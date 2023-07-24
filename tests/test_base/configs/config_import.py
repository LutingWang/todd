if '_import_' not in globals():
    import importlib
    _import_ = importlib.import_module

torch = _import_('torch')  # noqa: F821
fsdp = _import_('torch.distributed.fsdp')  # noqa: F821
