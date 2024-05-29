from torch import nn

from todd.models.filters.named_module import NamedModulesFilter


class TestNamedModulesFilter:

    def test_named_members(self) -> None:
        filter_ = NamedModulesFilter()
        module = nn.Module()
        module.conv1 = nn.Conv2d(3, 64, 3)
        module.conv2 = nn.Conv2d(64, 64, 3)
        module.fc = nn.Linear(64, 10)
        result = filter_._named_members(module)
        assert set(result) == {
            ('', module),
            ('conv1', module.conv1),
            ('conv2', module.conv2),
            ('fc', module.fc),
        }
