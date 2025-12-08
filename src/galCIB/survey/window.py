"""
Contains radial window kernel calculating functions.
"""

import numpy as np
from scipy.integrate import simpson


def compute_Wcib(z):
    """
    Returns radial kernel of CIB.

    Note that we do NOT scale the output by the
    mean emissivity since the mean cancels out in
    C_ell calculation. For proper wcib, consider
    multiplying by mean emissivity.
    """

    wcib = 1 / (1 + z)

    return wcib


def compute_Wmu(z, pz, mag_alpha, cosmo):
    """
    Returns radial kernel of magnification bias.

    Args:
        z : redshift range
        pz : redshift distribution
        mag_alpha : magnification bias alpha param. as a func. of z
        cosmo: Cosmology object that helps with pre-factor
        calculations.
    """

    chi = cosmo.chi
    H0 = cosmo.H0
    c = cosmo.c
    Om0 = cosmo.Om0

    prefactor = 2 * 3 / 2 * Om0 * (H0 / c) ** 2 * (1 + z)

    # integral from z to zstar
    integral_term = np.zeros_like(z)
    for i in range(len(z)):
        chi_at_z = chi[i]
        mask_from_z_to_zstar = z >= z[i]
        chi_from_z_to_zstar = chi[mask_from_z_to_zstar]
        alpha_from_z_to_zstar = mag_alpha[mask_from_z_to_zstar]
        pz_from_z_to_zstar = pz[mask_from_z_to_zstar]

        integrand = (
            chi_at_z
            * (1 - chi_at_z / chi_from_z_to_zstar)
            * (alpha_from_z_to_zstar - 1)
            * pz_from_z_to_zstar
        )
        integral_term[i] = simpson(integrand, x=z[mask_from_z_to_zstar])

    wmu = prefactor * integral_term

    return wmu


def compute_Wg(pz, cosmo=None):
    """
    Returns radial kernel of galaxy survey.
    """

    wgal = pz * 1 / cosmo.dchi_dz

    return wgal
