import difflib
import pathlib
import tempfile

import pytest

from todd.base._extensions import setattr_temp
from todd.base.configs import Config


class TestConfigs:

    @pytest.fixture
    def config(self) -> Config:
        return Config(a=1)

    def test_merge(self, config: Config) -> None:
        assert Config.merge(1, 2) == 2
        assert Config.merge(config, 2) == 2

        assert Config.merge(1, Config(_delete_=True, b=2)) == Config(b=2)
        assert Config.merge(config, Config(_delete_=True, b=2)) \
            == Config(b=2)

        assert Config.merge(1, Config(b=2)) \
            == Config(b=2)

        assert Config.merge(Config(a=1, b=dict(c=3)), Config(b=dict(c='c'))) \
            == Config(a=1, b=dict(c='c'))

        assert Config.merge(
            Config(a=dict(b=dict(c=3))),
            Config(a=dict(_delete_=True, b=dict(d='d'))),
        ) == Config(a=dict(b=dict(d='d')))

    def test_load(self, data_dir: pathlib.Path) -> None:
        assert Config.loads('a = 1\nb = dict(c=3)') == Config(a=1, b=dict(c=3))
        assert Config.loads('a = 1\nb = dict(c=v)', globals=dict(v=3)) \
            == Config(a=1, b=dict(c=3))
        assert Config.load(data_dir / 'config.py') == Config(a=1, b=dict(c=3))

    def test_diff(self, data_dir: pathlib.Path) -> None:
        a = Config.load(data_dir / 'base.py')
        b = Config.load(data_dir / 'config.py')
        html = data_dir / 'diff.html'
        assert Config.diff(a, b) == ('  a = 1\n'
                                     '+ b = dict(c=3)\n'
                                     '  ')
        with setattr_temp(difflib.HtmlDiff, '_default_prefix', 0):
            assert Config.diff(a, b, 'html') == html.read_text()
        with pytest.raises(ValueError, match='custom_mode'):
            Config.diff(a, b, 'custom_mode')

    def test_dump(self) -> None:
        config = Config(
            a=1,
            b=dict(c=3),
            d={
                5: 'e',
                'f': ['g', ('h', {'i', 'j'})],
            },
        )
        with tempfile.NamedTemporaryFile() as f:
            config.dump(f.name)
            assert Config.load(f.name) == config
