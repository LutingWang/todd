if '_import_' not in globals():
    import importlib
    _import_ = importlib.import_module

torch = _import_('torch')
fsdp = _import_('torch.distributed.fsdp')
