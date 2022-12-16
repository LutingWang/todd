import logging
import pathlib
import re

import pytest

from todd import Store
from todd.base.loggers import SGR, get_logger


class TestSGR:

    def test_format(self) -> None:
        assert SGR.format(
            'hello',
            (SGR.UNDERLINE, SGR.FG_GREEN, SGR.BG_GREEN),
        ) == '\033[4;32;42mhello\033[m'


class TestGetLogger:

    @pytest.mark.usefixtures('teardown_logger')
    @pytest.mark.parametrize('logger_name', [__name__])
    def test_get_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger()
        assert logger.name == __name__

        logger.propagate = True
        with caplog.at_level(logging.INFO, logger=__name__):
            logger.info('hello')
        logger.propagate = False

        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].message == 'hello'

    @pytest.mark.usefixtures('teardown_logger')
    @pytest.mark.parametrize('logger_name', [__name__])
    def test_log_file(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / 'log.txt'
        Store.LOG_FILE = str(log_file)

        logger = get_logger()
        logger.info('hello')

        assert log_file.exists()
        with log_file.open() as f:
            assert any(re.search('INFO.*hello', line) for line in f)

        Store.LOG_FILE = ''
