import difflib
import pathlib

import pytest

from todd import FileType, set_temp
from todd.base.configs import Config


class TestConfigs:

    @pytest.fixture
    def config(self) -> Config:
        return Config(a=1, b=dict(c=3))

    def test_load(self, config: Config, data_dir: pathlib.Path) -> None:
        assert Config.load(data_dir / 'config.py') == config

    def test_diff_html(self, config: Config, data_dir: pathlib.Path) -> None:
        diff = Config(a=1).diff(config, FileType.HTML)
        html = data_dir / 'diff.html'
        with set_temp(difflib.HtmlDiff, '._default_prefix', 0):
            assert diff == html.read_text()
