import difflib
import pathlib

import pytest

from todd.configs.base import Config
from todd.utils import set_temp


class TestConfigs:

    @pytest.fixture
    def config(self) -> Config:
        return Config(a=1, b=dict(c=3))

    def test_load(self, config: Config, data_dir: pathlib.Path) -> None:
        assert Config.load(data_dir / 'config.py') == config

    def test_load1(self, data_dir: pathlib.Path) -> None:
        assert Config.load(data_dir / 'config1.py') == Config(a=dict(c=3))

    def test_load2(self, data_dir: pathlib.Path) -> None:
        assert Config.load(data_dir / 'config2.py') == Config(a=[dict(b=3)])

    def test_load2_1(self, data_dir: pathlib.Path) -> None:
        assert Config.load(data_dir / 'config2_1.py') == \
            Config(a=[dict(b=2), dict(c=3)])

    def test_diff_html(self, config: Config, data_dir: pathlib.Path) -> None:
        diff = Config(a=1).diff(config, True)
        html = data_dir / 'diff.html'
        with set_temp(difflib.HtmlDiff, '._default_prefix', 0):
            assert diff == html.read_text()

    def test_load_import(self, data_dir: pathlib.Path) -> None:
        config_import = Config.load(data_dir / 'config_import.py')
        assert config_import.dumps() == \
            '''fsdp = _import_('torch.distributed.fsdp')
torch = _import_('torch')
TYPE_CHECKING = False
'''
