"""
Contains class that handles inverse Fourier Transform
of the satellite profile inside the halo.
"""

import numpy as np
import warnings

from .default_models import compute_unfw, compute_nfw_exp_mixed
from .utils import (
    _compute_default_r_delta,
    _compute_default_concentration,
    _compute_r_star,
    _compute_nu,
    _compute_delta_halo,
)

warnings.simplefilter("once")


class SatProfile:
    """
    Satellite galaxy profile model. Computes u(k, M, z).

    Args:
        cosmology: Cosmology object with .k, .Mh, .z, and .rho_m
        theta: Optional parameter vector if profile or concentration depends on it
        profile_model: either 'nfw' or a custom callable (k, M, z, c) -> u
    """

    def __init__(
        self, cosmology, theta=None, c=None, rs=None, profile_type="nfw", delta_h=200
    ):
        self.cosmo = cosmology
        self.Mh = cosmology.Mh  # (k,M,z)
        self.z = cosmology.z  # (k,M,z)
        self.k_grid = cosmology.k_grid  # (k,M,z)
        self.theta = theta  # for future model extension
        self.dlnpk_dlnk = cosmology.dlnpk_dlnk  # concentration calculation
        self.profile_type = profile_type
        self.delta_h = delta_h

        if self.profile_type == "mixed" and self.theta is not None:
            warnings.warn(
                "For the default mixed profile, theta should be [f_exp, tau_exp, lambda_NFW].",
                category=UserWarning,
            )

        # Compute or load concentration and rstar parameters
        if c is None or rs is None:
            self._cache_params()  # load relevant params
        else:
            self.c = c
            self.rs = rs

        # initialize u profile
        self.u = self._compute_u_profile()

        # initialize halo bias
        self._compute_bnu()

    def _cache_params(self):
        """
        Initializes concentration and rs parameters
        """

        self.r200 = _compute_default_r_delta(self.Mh, self.cosmo, self.delta_h)
        self.c = _compute_default_concentration(self.r200, self.cosmo)
        self.rs = _compute_r_star(self.r200, self.c)
        self.nu = _compute_nu(self.cosmo)

    def _compute_u_profile(self):
        if self.profile_type == "nfw":
            u = compute_unfw(self.k_grid, self.c, self.rs, lambda_NFW=1)
        elif self.profile_type == "mixed":
            u = compute_nfw_exp_mixed(self.k_grid, self.c, self.rs, self.theta)

        return u

    def get_u_profile(self):
        return self.u

    def update_theta(self, theta):
        """
        Update model parameters and recompute the profile u(k, M, z).
        Only needed for theta-dependent models (e.g., 'mixed').
        """
        self.theta = theta
        self._cache_params()
        self.u = self._compute_u_profile()

    # def _compute_halo_bias(self, model=None):
    #     """
    #     Returns halo bias using colossus; defaults to Tinker10
    #     """

    #     self.hbias = bias.haloBiasFromNu(self.nu, self.z, mdef='200m')

    def _compute_bnu(self):
        """
        Returns halo bias based on Tinker+2010.
        """

        # Can also use "crit" for w.r.t. critical density
        delta = _compute_delta_halo(self, delta_wrt="mean")

        y = np.log10(delta)
        A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
        aa = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4))
        c = 2.4
        nuu = self.nu
        dc = 1.686  # neglecting the redshift evolution

        self.bnu = 1 - (A * nuu**aa / (nuu**aa + dc**aa)) + B * nuu**b + C * nuu**c
