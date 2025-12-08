import os

from galCIB.io import load_my_filters


FILTERS_DIR = os.path.join(".", "data", "minimal", "filters")


class TestFunctionalLoadMyFilters:
    def test_load_filters(self):
        filters = load_my_filters(FILTERS_DIR)
        assert filters[100]
        assert filters[143]
        assert filters[100][0].shape == (12297,)
        assert filters[100][1].shape == (12297,)
        assert filters[100][0][500] == 305788301000.0
        assert filters[143][0][500] == 206856799000.0
