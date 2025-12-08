import pytest

from galCIB.utils import get_color_correction


class TestColorCorrection:
    @pytest.mark.parametrize("nu, expected", [(100, 1.076), (857, 0.995)])
    def test_can_get_color_correction(self, nu, expected):
        result = get_color_correction(nu)
        assert result == expected

    def test_raises_value_error_for_unknown_nu(self):
        with pytest.raises(ValueError, match="Do not have a color"):
            get_color_correction(1)
