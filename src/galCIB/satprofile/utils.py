"""
Utility file with commonly used functions.
"""

import numpy as np
from scipy.integrate import simpson


def _compute_default_concentration(r200, cosmo):
    """
    Returns

    Args:
        r200 : 200 times overdensity radius
        cosmo : Cosmology object
    Returns:
        res : (Mh, z)
    """

    use_mean = False  # 2 relations provided. mean and median.
    phi0_median, phi0_mean = 6.58, 7.14
    phi1_median, phi1_mean = 1.37, 1.60
    eta0_median, eta0_mean = 6.82, 4.10
    eta1_median, eta1_mean = 1.42, 0.75
    alpha_median, alpha_mean = 1.12, 1.40
    beta_median, beta_mean = 1.69, 0.67
    _nu = _nu_delta(r200, cosmo)  # shape (Mh, z)
    n_k = _interpolate_grad_over_kR(cosmo)[:, np.newaxis]  # shape (Mh, z)

    if use_mean:
        c_min = phi0_mean + phi1_mean * n_k
        nu_min = eta0_mean + eta1_mean * n_k
        res = c_min * ((_nu / nu_min) ** -alpha_mean + (_nu / nu_min) ** beta_mean) / 2
    else:
        c_min = phi0_median + phi1_median * n_k
        nu_min = eta0_median + eta1_median * n_k
        res = (
            c_min
            * ((_nu / nu_min) ** -alpha_median + (_nu / nu_min) ** beta_median)
            / 2
        )

    return res


def _interpolate_grad_over_kR(cosmo):
    """
    Returns interpolated dlnpk_dlnk based on
    the wavenumber corresponding to Lagrangian radii
    of the halo masses.
    """

    kr = _compute_k_R(cosmo)
    dlnpk_dlnk_interped = np.interp(kr, cosmo.k, cosmo.dlnpk_dlnk)
    return dlnpk_dlnk_interped


def _compute_k_R(cosmo):
    """
    Returns Fourier Transform wavenumber corresponding to
    Lagrangian Radii of Haloes.

    We need a c-M relation to calculate the fourier transform of the NFW
    profile. We use the relation from https://arxiv.org/pdf/1407.4730.pdf
    where they use the slope of the power spectrum with respect to the
    wavenumber in their formalism. This power spectrum slope has to be
    evaluated at a certain value of k such that k = kappa*2*pi/rad
    This kappa value comes out to be 0.69 according to their calculations.
    """
    rad = _compute_mass_to_radius(cosmo)
    kappa = 0.69
    return kappa * 2 * np.pi / rad


def _compute_mass_to_radius(cosmo):
    """
    Returns Lagrangian radius of a dark matter halo
    """
    # rho_mean = self.mean_density0()

    rho_mean = cosmo.mean_density0
    r3 = 3 * cosmo.Mh / (4 * np.pi * rho_mean)

    return r3 ** (1.0 / 3.0)


def _compute_default_r_delta(Mh, cosmology, delta_h=200):
    """
    Returns radius of the halo containing amount of matter
    corresponding to delta times the critical density of
    the universe inside that halo.

    Args:
        delta_h : How many times the critical density to consider to define halo. Default 200.
        rho_crit : Cosmic critical density in units of Msol/Mpc^3
    Returns:
        r : radius in units of Mpc
    """

    r3 = (
        3
        * Mh[:, np.newaxis]
        / (4 * np.pi * delta_h * cosmology.rho_crit[np.newaxis, :])
    )  # [Mpc]^3
    r = r3 ** (1 / 3)

    return r


def _nu_delta(rad, cosmo):
    """
    Returns the size of the peak heights given radius.

    We use r_delta rather than the simple Lagrangian radius.
    This will be used in c-M relation to calculate the NFW profile.
    """

    delta_c = 1.686  # critical density of the universe. Redshift evolution is small and neglected
    sig = _sigma(rad, cosmo)  # shape (Mh, z)
    return delta_c / sig


def _compute_nu(cosmo):
    """
    Returns peak heights.

    Simple Lagrangian radius calculated using mass_to_radius function.
    """

    rad = _compute_mass_to_radius(cosmo)
    delta_c = 1.686  # critical density of the universe. Redshift evolution is small and neglected
    sig = _sigma(rad, cosmo)
    return delta_c / sig  # length of mass array


def _sigma(rad, cosmo):
    """
    Vectorized RMS matter fluctuation σ(R) over (k, M, z) grid.

    Args:
        rad : ndarray of shape (Nm,) or (Nm, Nz)
        cosmo : Cosmology object with .k and .pk_grid (Nk, Nz)

    Returns:
        sigma : ndarray of shape (Nm, Nz)
    """
    k = cosmo.k  # shape (Nk,)
    lnk = np.log(k)
    pk = cosmo.pk_grid  # shape (Nk, Nz)

    if rad.ndim == 1:
        rad = rad[:, np.newaxis]  # shape (Nm, 1), broadcastable
    elif rad.ndim != 2:
        raise ValueError("rad must be 1D or 2D")

    rk = rad[..., np.newaxis] * k  # shape (Nm, Nz, Nk)
    Wrk = _W(rk)  # shape (Nm, Nz, Nk)
    rest = pk.T[np.newaxis, :, :] * k[np.newaxis, np.newaxis, :] ** 3  # (1, Nz, Nk)

    integrand = rest * Wrk**2  # shape (Nm, Nz, Nk)
    integ = simpson(integrand, x=lnk, axis=-1)  # shape (Nm, Nz)

    return np.sqrt((0.5 / np.pi**2) * integ)


def _W(rk):
    """
    Returns Fourier transform of top hat window function.

    Limit of 1.4e-6 put as it doesn't add much to the final answer and
    helps for faster convergence
    """

    res = np.where(rk > 1.4e-6, (3 * (np.sin(rk) - rk * np.cos(rk)) / rk**3), 1)

    return res


def _compute_r_star(r200, c200c):
    """
    Characteristic radius also called r_s in other literature.
    Physically refers to the transition point where the halo
    goes from a more concentrated inner region to a more
    diffuse outer region.

    $ c \equiv \frac{r_{200}{r_s}$

    Returns:
        res : (Mh, z)
    """

    rstar = r200 / c200c

    return rstar


def _compute_delta_halo(prof, delta_wrt="mean"):
    """Overdensity of a halo w.r.t
    mean density or critical density"""

    if delta_wrt == "mean":
        return prof.delta_h

    elif delta_wrt == "crit":
        return prof.delta_h / prof.cosmo.Om_z
