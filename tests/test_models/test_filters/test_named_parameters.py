from torch import nn

from todd.models.filters.named_parameters import NamedParametersFilter


class TestNamedParametersFilter:

    def test_named_members(self) -> None:
        filter_ = NamedParametersFilter()
        module = nn.Module()
        module.conv = nn.Conv2d(3, 64, 3)
        result = filter_._named_members(module)
        assert set(result) == {
            ('conv.weight', module.conv.weight),
            ('conv.bias', module.conv.bias),
        }
