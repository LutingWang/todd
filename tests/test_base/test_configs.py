import pathlib

import pytest

from todd.base.configs import Config


class TestConfigs:

    @pytest.fixture
    def config(self) -> Config:
        return Config(a=1)

    def test_setitem(self, config: Config) -> None:
        config['b'] = dict(c=3)
        assert isinstance(config['b'], Config)

    def test_attrs(self, config: Config) -> None:
        assert config.a == 1
        with pytest.raises(AttributeError, match='b'):
            config.b

        config.b = dict(c=3)
        assert config.b == Config(c=3)
        assert config.b.c == 3  # type: ignore[attr-defined]

        del config.b
        assert 'b' not in config

        config.__dict__['_privilege'] = True

        config.b = 2
        assert 'b' in config.__dict__

        del config.b
        assert 'b' not in config.__dict__

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

    def test_fromfile(self, data_dir: pathlib.Path) -> None:
        assert Config.fromfile(data_dir / 'config.py') == Config(a=1, b=2)
