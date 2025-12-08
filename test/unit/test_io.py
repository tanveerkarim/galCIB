from unittest.mock import patch

import pytest

from galCIB.io import load_my_filters


class TestLoadMyFilters:
    @pytest.mark.xfail(reason="TODO")
    def test_raises_value_error_if_directory_not_found(self):
        assert False

    @patch("galCIB.io.os.listdir", return_value=[])
    def test_returns_empty_dict_if_no_filters(self, mock_listdir):
        result = load_my_filters("pattern")
        assert result == {}

    @pytest.mark.xfail(reason="TODO")
    def test_returns_all_matched_filters(self):
        assert False

    @pytest.mark.xfail(reason="TODO")
    def test_returns_subset_of_matched_filters(self):
        assert False

    @pytest.mark.xfail(reason="TODO")
    def test_returns_tuple_of_Hz_and_normalized_response(self):
        assert False
