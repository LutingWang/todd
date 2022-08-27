import logging
import pathlib
import re

import pytest

from todd.base._extensions.logging import (
    ANSI,
    SGR,
    get_log_file,
    get_logger,
    init_log_file,
    log_file_initialized,
)


class TestANSI:

    def test_to_str(self) -> None:
        assert ANSI.to_str(False) == '0'
        assert ANSI.to_str(True) == '1'
        assert ANSI.to_str(2) == '2'
        assert ANSI.to_str(3.0) == '3'


class TestSGR:

    def test_to_str(self) -> None:
        assert SGR.to_str('NORMAL') == '0'
        assert SGR.to_str('bold') == '1'
        assert SGR.to_str('Faint') == '2'
        assert SGR.to_str(SGR.FG_BLACK) == '30'

    def test_format(self) -> None:
        assert SGR.format('hello') == '\033[mhello\033[m'
        assert SGR.format('hello', (0, )) == '\033[0mhello\033[m'
        assert SGR.format(
            'hello', ('ITALIC', 'FG_RED', 'BG_RED')
        ) == '\033[3;31;41mhello\033[m'
        assert SGR.format(
            'hello', (SGR.UNDERLINE, SGR.FG_GREEN, SGR.BG_GREEN)
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
        init_log_file(log_file)
        assert log_file_initialized()
        assert get_log_file() == log_file

        logger = get_logger()
        logger.info('hello')

        assert log_file.exists()
        with log_file.open() as f:
            assert any(re.search('INFO.*hello', line) for line in f)

        init_log_file(None)
        assert not log_file_initialized()
