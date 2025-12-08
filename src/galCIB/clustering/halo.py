"""
This module contains functions important for halo model calculations.
"""

import numpy as np
import scipy.special as ss
from scipy.integrate import simpson

from colossus.cosmology import cosmology as cc
from colossus.lss import bias as clss_bias

# import local modules
from .. import consts
from .cosmology import core

# import halo consts
Mh = consts.Mhc_Msol  # FIXME: make sure passing the correct form, Mhc or Mh
kk = consts.k_grid_over_ell  # (k, z)
power = consts.Pk_array_over_ell  # (z, k)


def uprof_mixed(prof_params, rad200, c, c_term, plot=False):
    """
    Returns Fourier Transform of halo profile mimicing DESI ELGs.

    According to 2306.06319, the ELG halo profile is:
    p(r|M) = f_exp * exp(-r/(r_s * tau)) * (1-fexp)[NFW](r|M).

    Args:
        prof_params: profile parameters passed to function
            fexp : frac. of ELG satellites following exp. profile
            rs :
            tau : exponential index
    Returns:
        res : (k, Mh, z)
    """

    fexp, tau, lambda_NFW = prof_params

    nfw_term = nfwfourier_u(lambda_NFW, rad200, c, c_term)
    exp_term = expfourier_u(tau=tau, rs=rad200)

    res = fexp * exp_term + (1 - fexp) * nfw_term

    if plot:  # only to plot
        return res, exp_term, nfw_term
    else:
        return res


def expfourier_u(tau, rs):
    """
    Returns Fourier Transform of the exp. decay profile.

    Args:
        tau : index of the exponential slope
        lambda_NFW : extending the NFW profile cut-off factor
    Returns:
        res : (k, Mh, z)
    """

    rs_times_tau = rs * tau  # (Mh, z)
    rs_times_tau = np.expand_dims(rs_times_tau, axis=0)  # (k, Mh, z)
    denom = ((rs_times_tau) ** 2 * kk[:, np.newaxis, :] ** 2 + 1) ** 2
    res = 1 / denom

    return res


def nfwfourier_u(lambda_NFW, rad200, c, c_term):
    """
    Returns Fourier Transform of the NFW profile.

    From 2.14 of 2310.10848.
    Form is: [ln(1+c) - c/(1+c)]^(-1) [cos(q)(Ci(q + cq) - Ci(q)) + sin(q)(Si(q + cq) - Si(q)) - sin(cq)/(1 + cq)]

    Note rs is rescaled as rs/lambda_NFW according to pg 24 of 2306.06319.
    This applies for ELG-type haloes.

    Args:
        lambda_NFW : rescaling factor of rs
    """

    # rs_original = r_star(rad200, c)
    rs_rescaled = r_star(
        rad200 / lambda_NFW, c
    )  # FIXME: does concentration change with rescaling?
    # FIXME: calculate q a priori and just divide by lambda to get q_rescaled
    q = kk[:, np.newaxis, :] * rs_rescaled[np.newaxis, :, :]  # (k, Mh, z)

    # broadcast to match q
    c = np.expand_dims(
        c, axis=0
    )  # FIXME: calculate c*q and only divide by lambda to get c*q rescaled
    c_term = np.expand_dims(c_term, axis=0)
    # tst = q + c*q

    Si_qcq, Ci_qcq = sine_cosine_int(q + c * q)  # (k, Mh, z)
    Si_q, Ci_q = sine_cosine_int(q)  # (k, Mh, z)

    cos_q_term = np.cos(q) * (Ci_qcq - Ci_q)
    sin_q_term = np.sin(q) * (Si_qcq - Si_q)
    sin_qc_term = np.sin(c * q) / (q + c * q)

    unfw = c_term * (cos_q_term + sin_q_term - sin_qc_term)  # (k, Mh, z)

    # return 0
    return unfw


# TODO: This is defined twice in this file
def ampl_nfw(c):
    """
    Dimensionless amplitude of the NFW profile.
    Gives:
        $\frac{1}{\log(1+c) - \frac{c}{1+c}}$
    """
    return 1.0 / (np.log(1.0 + c) - c / (1.0 + c))


def sine_cosine_int(x):
    """
    sine and cosine integrals required to calculate the Fourier transform
    of the NFW profile.
    $ si(x) = \int_0^x \frac{\sin(t)}{t} dt \\
    ci(x) = - \int_x^\infty \frac{\cos(t)}{t}dt$
    """

    si, ci = ss.sici(x)
    return si, ci


def r_star(r200, c_200c):  # , dlnpk_dlnk):
    """
    Characteristic radius also called r_s in other literature.
    Physically refers to the transition point where the halo
    goes from a more concentrated inner region to a more
    diffuse outer region.

    $ c \equiv \frac{r_{200}{r_s}$

    Returns:
        res : (Mh, z)
    """

    # c_200c = nu_to_c200c(rad, dlnpk_dlnk) # (Mh, z)
    # r200 = r_delta(delta_h=200) # (Mh, z)
    res = r200 / c_200c

    return res


# FIXME: citation for this?
def nu_to_c200c(rad, dlnpk_dlnk):  # length of mass array
    """
    Returns

    Args:

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
    _nu = nu_delta(rad)  # shape (Mh, z)
    n_k = dlnpk_dlnk  # shape (Mh, z)

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


def mass_to_radius():
    """
    Lagrangian radius of a dark matter halo
    """

    rho_mean = consts.mean_density0
    r3 = 3 * Mh / (4 * np.pi * rho_mean)
    return r3 ** (1.0 / 3.0)


def nu():
    """
    Calculate the peak heights: nu, we use the
    simple Lagrangian radius calculated using mass_to_radius function.

    Returns:
        nu : (Mh, z)
    """

    rad = mass_to_radius()
    delta_c = 1.686  # critical density of the universe. Redshift evolution
    # is small and neglected
    sig = sigma(rad)
    return delta_c / sig  # length of mass array


# DONE
def nu_delta(rad):  # peak heights
    """
    Returns the size of the peak heights.

    We use r_delta rather than the simple Lagrangian radius.
    This will be used in c-M relation to calculate the NFW profile.

    Args:
        rad: #FIXME: what is it?
    """

    delta_c = 1.686  # critical density of the universe. Redshift evolution is small and neglected #FIXME: what is this?
    sig = sigma(rad)  # shape (Mh, z)
    return delta_c / sig


# DONE
def sigma(rad):
    """
    Returns matter variance for given power spectrum,
    wavenumbers k and radius of the halo.
    Radius is calculated below from mass of the halo.
    It has to be noted that there's no factor
    of \Delta = 200 to calculate the radius.

    Args:
        rad: radius of variance size
    Returns:
        res: RMS at size of radius rad shape (Mh, z)
    """

    # need the full Pk to get the proper numerical integration
    kk_full = consts.Plin["k"]  # (k,)
    pk_full = consts.Plin["pk"].T  # (k,z)

    if rad.ndim == 1:
        rk = (
            rad[np.newaxis, :, np.newaxis] * kk_full[:, np.newaxis, np.newaxis]
        )  # (k, Mh, z)
    else:
        rk = rad[np.newaxis, :, :] * kk_full[:, np.newaxis, np.newaxis]  # (k, Mh, z)
    rest = pk_full * kk_full[:, np.newaxis] ** 3  # shape (k, z)
    lnk = np.log(kk_full)  # shape (k,)
    Wrk = W(rk)  # (k, Mh, z)
    integ = rest[:, np.newaxis, :] * Wrk**2  # (k,Mh,z)
    sigm = (0.5 / np.pi**2) * simpson(
        integ, x=lnk[:, np.newaxis, np.newaxis], axis=0
    )  # integrating along kk
    res = np.sqrt(sigm)

    return res  # , rk


# DONE
def W(rk):
    """
    Returns Fourier Transform of top-hat window function.

    Limit of 1.4e-6 put as it doesn't add much to the final answer and
    helps for faster convergence.
    """

    greater_term = 3 * (np.sin(rk) - rk * np.cos(rk)) / rk**3
    res = np.where(rk > 1.4e-6, greater_term, 1)

    return res


# DONE
def r_delta(delta_h=200):
    """
    Returns radius of the halo containing amount of matter
    corresponding to delta times the critical density of
    the universe inside that halo.

    Args:
        delta_h : How many times the critical density to consider to define halo. Default 200.
        rho_crit : Cosmic critical density in units of Msol/Mpc^3, shape (z,)
    Returns:
        res : shape (Mh, z) in units of Mpc^3
    """

    r3 = (
        3
        * consts.Mh_Msol[:, np.newaxis]
        / (4 * np.pi * delta_h * consts.rho_crit[np.newaxis, :])
    )  # units of Mpc^3

    res = r3 ** (1.0 / 3.0)
    return res


# TODO: This is defined twice in this file
# FIXME: do we infer for c?
def ampl_nfw_TWO(c):
    """
    Returns dimensionless amplitude of the NFW profile.

    Equation from 2.14 of 2310.10848.
    Form: [ln(1 + c) - c/(1+c)]^(-1)

    Args:
        c : concentration parameter
    Returns:
        a : amplitude value
    """

    a = 1.0 / (np.log(1.0 + c) - c / (1.0 + c))
    return a


def get_dlnpk_dlnk():
    """
    When the power spectrum is obtained from CAMB, slope of the ps wrt k
    shows wiggles at lower k which corresponds to the BAO features. Also
    at high k, there's a small dip in the slope of the ps which is due to
    the effect of the baryons (this is not very important for the current
    calculations though). We are using the analysis from the paper
    https://arxiv.org/pdf/1407.4730.pdf where they have used the power
    spectrum from Eisenstein and Hu 1998 formalism where these effects have
    been negelected and therefore they don't have the wiggles and the bump.
    In order to acheive this, we have to smooth out the
    slope of the ps at lower k. But we have checked that the results
    do not vary significantly with the bump.

    Returns:
        res : (Mh, z)
    """
    pk_all = consts.Plin["pk"]
    k_all = consts.Plin["k"]
    grad = np.zeros((len(pk_all), len(k_all)))  # shape (z,k)

    grad[:, :-1] = np.diff(np.log(pk_all), axis=1) / np.diff(np.log(k_all), axis=0)
    # FIXME: is this not just copying the last value again?
    grad[:, -1] = (np.log(pk_all[:, -1]) - np.log(pk_all[:, -2])) / (
        np.log(k_all[-1]) - np.log(k_all[-2])
    )
    kr = consts.k_R  # (Mh,)

    res = np.zeros((len(kr), pk_all.shape[0]))  # shape(Mh, z)
    for i in range(pk_all.shape[0]):  # loop over redshift
        res[:, i] = np.interp(kr, k_all, grad[i, :])

    return res


##--BIAS MODEL--##
def b_nu(nu, z):
    """
    Returns halo bias at a given peak height nu.
    """

    # set cosmology for colossus
    cc.setCurrent(core.colossus_planck_cosmo)

    b = clss_bias.haloBiasFromNu(
        nu, z, mdef="200m"
    )  # calculate bias for both ELG and CIB

    return b
