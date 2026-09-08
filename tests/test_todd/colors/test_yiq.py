import pytest

from todd.colors import RGB
from todd.colors.yiq import YIQ


class TestYIQ:

    def test_init(self) -> None:
        yiq = YIQ(1, 0, 0)

        assert yiq.luminance == 1

        with pytest.raises(AssertionError):
            YIQ(255, 0, 0)

    def test_to_tuple(self) -> None:
        yiq = YIQ(.5, -.25, .125)

        assert yiq.to_tuple() == (.5, -.25, .125)

    def test_from(self) -> None:
        gray = RGB(128 / 255, 128 / 255, 128 / 255)

        yiq = gray.to(YIQ)

        assert yiq.luminance == pytest.approx(128 / 255)
        assert yiq.to(RGB).to_tuple(normalized=True) == pytest.approx(
            gray.to_tuple(normalized=True),
        )
