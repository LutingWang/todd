import pytest

from todd.colors import RGB
from todd.colors.yiq import YIQ


class TestYIQ:

    def test_init(self) -> None:
        yiq = YIQ(luminance=1, in_phase=0, quadrature=0)

        assert yiq.luminance == 1

        with pytest.raises(AssertionError):
            YIQ(luminance=255, in_phase=0, quadrature=0)

    def test_to_tuple(self) -> None:
        yiq = YIQ(luminance=.5, in_phase=-.25, quadrature=.125)

        assert yiq.to_tuple() == (.5, -.25, .125)

    def test_from(self) -> None:
        gray = RGB(
            red=128 / 255,
            green=128 / 255,
            blue=128 / 255,
        )

        yiq = gray.to(YIQ)

        assert yiq.luminance == pytest.approx(128 / 255)
        assert yiq.to(RGB).to_tuple(normalized=True) == pytest.approx(
            gray.to_tuple(normalized=True),
        )
