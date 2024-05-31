from typing import cast

import torch
from custom_types import CustomModule
from torch import nn

from todd import Config
from todd.registries.patches import OptimizerRegistry


class TestOptimizerRegistry:

    def test_params(self, model: CustomModule) -> None:
        config = Config(
            params=dict(type='NamedParametersFilter', regex='conv.[^w]'),
        )
        config = OptimizerRegistry.params(model, config)
        assert len(config) == 1
        assert len(config['params']) == 1
        assert config['params'][0] is cast(nn.Conv2d, model.conv).bias

    def test_build(self, model: CustomModule) -> None:
        config = Config(model=model)
        optimizer = OptimizerRegistry._build(torch.optim.Adam, config)
        assert isinstance(optimizer, torch.optim.Adam)
        assert set(optimizer.param_groups[0]['params']) == \
            set(model.parameters())
