from galCIB.cib.default_snu import get_snu_factory, snu_Y23_factory, snu_M21_factory


class TestGetSnuFactory:
    def test_get_snu_factory(self):
        assert get_snu_factory("Y23") == snu_Y23_factory
        assert get_snu_factory("M21") == snu_M21_factory
