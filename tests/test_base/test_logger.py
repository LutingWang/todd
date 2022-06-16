import logging
import pathlib
import re

import pytest

from todd.base._extensions.logging import ANSI, SGR, get_logger


class TestANSI:

    def test_to_str(self):
        assert ANSI.to_str(False) == '0'
        assert ANSI.to_str(True) == '1'
        assert ANSI.to_str(2) == '2'
        assert ANSI.to_str(3.0) == '3'


class TestSGR:

    def test_to_str(self):
        assert SGR.to_str('NORMAL') == '0'
        assert SGR.to_str('bold') == '1'
        assert SGR.to_str('Faint') == '2'
        assert SGR.to_str(SGR.FG_BLACK) == '30'

    def test_format(self):
        assert SGR.format('hello') == '\033[mhello\033[m'
        assert SGR.format('hello', (0, )) == '\033[0mhello\033[m'
        assert SGR.format(
            'hello', ('ITALIC', 'FG_RED', 'BG_RED')
        ) == '\033[3;31;41mhello\033[m'
        assert SGR.format(
            'hello', (SGR.UNDERLINE, SGR.FG_GREEN, SGR.BG_GREEN)
        ) == '\033[4;32;42mhello\033[m'


class TestGetLogger:

    @pytest.fixture()
    def teardown_logger(self):
        yield
        logger = logging.Logger.manager.loggerDict.pop(__name__)
        for handler in logger.handlers:
            handler.close()

    @pytest.mark.usefixtures('teardown_logger')
    def test_get_logger(self, caplog: pytest.LogCaptureFixture):
        logger = get_logger()
        assert logger.name == __name__

        logger.propagate = True
        with caplog.at_level(logging.INFO, logger=__name__):
            logger.info('hello')
        logger.propagate = False

        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].message == 'hello'

    @pytest.mark.usefixtures('teardown_logger')
    def test_get_logger_with_log_file(self, tmp_path: pathlib.Path):
        log_file = tmp_path / 'log.txt'
        logger = get_logger(log_file)
        assert logger.name == __name__

        logger.info('hello')

        assert log_file.exists()
        with log_file.open() as f:
            assert any(re.search('INFO.*hello', line) for line in f)
