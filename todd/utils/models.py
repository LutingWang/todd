from typing import Dict, Optional

from ..base import Module, Registry, getattr_recur


class ModelLoader:

    @staticmethod
    def load_mmlab_models(
        registry: Registry[Module],
        config: str,
        config_options: Optional[str] = None,
        ckpt: Optional[str] = None,
    ) -> Module:
        import mmcv
        from mmcv.runner import load_checkpoint

        config_dict = mmcv.Config.fromfile(config)
        if config_options is not None:
            config_dict.merge_from_dict(config_options)
        model = registry.build(config_dict.model)
        if ckpt is not None:
            load_checkpoint(model, ckpt, map_location='cpu')
            model._is_init = True
        return model

    @staticmethod
    def load_state_dict(target: Module, source: Module):
        state_dict = source.state_dict()
        missing_keys, unexpected_keys = target.load_state_dict(
            state_dict, strict=False,
        )
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(missing_keys, unexpected_keys)  # TODO: enrich output
        target._is_init = True

    @staticmethod
    def load_state_dicts(models, prefixes: Dict[str, str]):
        for target, source in prefixes.items():
            target_module = getattr_recur(models, target)
            source_module = getattr_recur(models, source)
            ModelLoader.load_state_dict(target_module, source_module)
