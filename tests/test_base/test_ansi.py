from todd._patches.logging import ANSI, SGR


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
