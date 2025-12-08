# survey/survey.py

"""
This module sets up the galaxy survey specifications used in
the analysis.
"""

import numpy as np
from .window import compute_Wcib, compute_Wg, compute_Wmu


class Survey:
    def __init__(
        self,
        z,
        pz=None,
        mag_alpha=None,  # galaxy-specific
        cib_filters=None,  # dict: freq_GHz -> (freq_array_Hz, response_array)
        nu_obs=None,  # optional override of effective frequencies
        ells=None,
        nside=None,
        binned_ell_ledges=None,  # if Cl should be binned
        name=None,
    ):
        """
        Container Class for Survey.

        Args:
            z : redshift array
            pz : galaxy redshift distribution
            mag_alpha : galaxy magnification bias parameter alpha
            nu_obs : effective CIB frequencies
            filt_response : CIB filter response curves
        """

        self.name = name or "unnamed"
        self.z = z
        self.z_ratio = z / (1 + z)

        # galaxy-specific
        self.pz = pz
        self.mag_alpha = self._standardize_mag_alpha(mag_alpha)

        # CIB-specific filters
        self.filters = cib_filters or {}

        # C_ell specific
        self.ells = ells
        self.binned_ell_ledges = binned_ell_ledges
        self.nside = nside

        # If user does not provide nu_obs explicitly, use keys of filters sorted
        if nu_obs is None and self.filters:
            self.nu_obs = sorted(self.filters.keys())
        else:
            self.nu_obs = nu_obs or []

    def _standardize_mag_alpha(self, mag_alpha):
        """
        Standardizes mag_alpha, whether a scalar or a vector
        """
        if np.isscalar(mag_alpha):
            return np.full_like(self.z, mag_alpha)
        else:
            mag_alpha = np.asarray(mag_alpha)
            if mag_alpha.shape != self.z.shape:
                raise ValueError(
                    f"mag_alpha shape {mag_alpha.shape} does not match z shape {self.z.shape}"
                )
            return mag_alpha

    def compute_windows(self, cosmology):
        self.Wcib = compute_Wcib(self.z)
        self.Wg = compute_Wg(self.pz, cosmo=cosmology)
        self.Wmu = compute_Wmu(self.z, self.pz, self.mag_alpha, cosmology)

    def get_filter_response(self, freq_GHz):
        """
        Returns (freq_array_Hz, response_array) for given effective freq in GHz.
        """
        return self.filters.get(freq_GHz, (None, None))
