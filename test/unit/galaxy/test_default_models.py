import numpy as np

from galCIB import Cosmology
from galCIB.galaxy.default_models import (
    Ncen_mHMQ,
    Nsat_ELG,
    Ncen_Z05,
    Nsat_Z05,
    get_hod_model,
)


class TestGetHODModel:
    def test_get_desi_model(self):
        zs = np.arange(0.05, 10.05, 0.1)
        ks = np.logspace(-3, 1, 500)
        Mh = np.logspace(7, 15, 100)
        cosmo = Cosmology(
            zs, ks, Mh, colossus_cosmo_name="planck18", use_little_h=False
        )
        model = get_hod_model("DESI-ELG", cosmo)
        assert model.name == "DESI-ELG"
        assert model._ncen_fn == Ncen_mHMQ
        assert model._nsat_fn == Nsat_ELG
        assert model.use_log10M_ncen
        assert not model.use_log10M_nsat
        assert not model.uses_z_ncen
        assert not model.uses_z_nsat

    def test_get_zheng_model(self):
        zs = np.arange(0.05, 10.05, 0.1)
        ks = np.logspace(-3, 1, 500)
        Mh = np.logspace(7, 15, 100)
        cosmo = Cosmology(
            zs, ks, Mh, colossus_cosmo_name="planck18", use_little_h=False
        )
        model = get_hod_model("Zheng05", cosmo)
        assert model.name == "Zheng05"
        assert model._ncen_fn == Ncen_Z05
        assert model._nsat_fn == Nsat_Z05
        assert not model.use_log10M_ncen
        assert not model.use_log10M_nsat
        assert model.uses_z_ncen
        assert model.uses_z_nsat
