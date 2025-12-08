# cib/utils.py

import numpy as np
from scipy.integrate import simpson


def SED_to_flux(sed, freq_sed, freq_filt, filt_response):
    """
    Returns the predicted flux per Planck channel of a given SED.

    Args:
        sed : (Nz, Nwv) array of SEDs
        wv_sed : (Nz, Nwv) array of wavelength grids for each SED
        fwv : (Nfwv,) common wavelength grid for filter response
        fresponse : (Nfwv,) filter response curve
    """
    # Ensure sed is 2D: (Nz, Nwv)
    sed = np.atleast_2d(sed)

    # Normalize the response curve
    norm = simpson(filt_response, x=freq_filt)

    # Interpolate and integrate in one list comprehension
    flux = np.array(
        [
            simpson(
                np.interp(freq_filt, w_row, s_row, left=0.0, right=0.0) * filt_response,
                x=freq_filt,
            )
            for w_row, s_row in zip(freq_sed, sed)
        ]
    )

    return flux / norm


def _compute_BAR_grid(cosmo, Mh=None, m=None):
    """
    Compute the Baryon Accretion Rate (BAR) grid.

    Parameters
    ----------
    cosmo : object
        Cosmology object with Mh, z, Om0, Ode0, Ob_z, Om_z
    Mh : array-like or None
         Optional halo mass grid. Useful if effective halo mass differs
    m : array-like or None
        Optional mass grid. If None, uses cosmo.Mh (shape: (Nm,))
        If provided, shape can be (Nm,) or (Nm, NMh)

    Returns
    -------
    BAR : ndarray
        Baryon accretion rate. Shape will broadcast to (Nm, Nz) or (Nm, NMh, Nz)
    """
    if (m is None) and (Mh is None):
        m = cosmo.Mh  # (Nm,)
    elif m is None:
        m = Mh

    z = cosmo.z  # (Nz,)

    # Ensure correct shape broadcasting
    m = np.atleast_2d(m)  # shape: (Nm, ...) or (Nm, NMh)
    # if m.shape[-1] == len(cosmo.z):
    #     # Already includes z axis
    #     raise ValueError("Mass grid 'm' should not include redshift axis")

    # Reshape for broadcasting with z
    m = m[..., np.newaxis]  # (..., 1)
    z = z[np.newaxis]  # (1, Nz)

    Om0 = cosmo.Om0
    Ode0 = cosmo.Ode0
    Ob_z = cosmo.Ob_z  # (Nz,)
    Om_z = cosmo.Om_z  # (Nz,)

    Msol = 1e12
    MGR = (
        46.1 * (m / Msol) ** 1.1 * (1 + 1.11 * z) * np.sqrt(Om0 * (1 + z) ** 3 + Ode0)
    )  # (..., Nz)
    BAR = MGR * (Ob_z / Om_z)  # [np.newaxis, :]  # broadcast to (..., Nz)

    return BAR
