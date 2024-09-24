import difflib
import pathlib

import pytest

from todd.configs.py_ import PyConfig
from todd.utils import set_temp


class TestPyConfig:

    @pytest.fixture
    def config(self) -> PyConfig:
        return PyConfig(a=1, b=dict(c=3))

    def test_load(self, config: PyConfig, data_dir: pathlib.Path) -> None:
        assert PyConfig.load(data_dir / 'config.py') == config

    def test_load1(self, data_dir: pathlib.Path) -> None:
        assert PyConfig.load(data_dir / 'config1.py') == PyConfig(a=dict(c=3))

    def test_load2(self, data_dir: pathlib.Path) -> None:
        assert PyConfig.load(data_dir / 'config2.py') == \
            PyConfig(a=[dict(b=3)])

    def test_load2_1(self, data_dir: pathlib.Path) -> None:
        assert PyConfig.load(data_dir / 'config2_1.py') == \
            PyConfig(a=[dict(b=2), dict(c=3)])

    def test_diff_html(self, config: PyConfig, data_dir: pathlib.Path) -> None:
        diff = PyConfig(a=1).diff(config, True)
        html = data_dir / 'diff.html'
        with set_temp(difflib.HtmlDiff, '._default_prefix', 0):
            assert diff == html.read_text()

    def test_load_export(self, data_dir: pathlib.Path) -> None:
        config_import = PyConfig.load(data_dir / 'config_export.py')
        assert config_import.dumps() == "module = 'torch'\n"
