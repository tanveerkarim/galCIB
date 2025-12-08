"""
Contains the default models.

1. NFW Profile
2. Mixed profile: f * NFW + (1-f) * Exponential Decay
"""

import numpy as np
from scipy.special import sici


def compute_unfw(k, c, rs, lambda_NFW=1):
    """
    Returns Fourier Transform of the NFW profile.

    From 2.14 of 2310.10848.
    Form is: [ln(1+c) - c/(1+c)]^(-1) [cos(q)(Ci(q + cq) - Ci(q)) + sin(q)(Si(q + cq) - Si(q)) - sin(cq)/(1 + cq)]

    Note rs is rescaled as rs/lambda_NFW according to pg 24 of 2306.06319.
    This applies for ELG-type haloes.

    Args:
        k : wavenumber
        c : concentration
        rs : scale radius
    """

    rs = rs / lambda_NFW  # rescaling according to 2306.06319 Sec. below Eq. 7.7
    q = k * rs
    one_plus_c = 1 + c
    one_plus_c_times_q = one_plus_c * q

    Si_q, Ci_q = sici(q)
    Si_one_plus_c_times_q, Ci_one_plus_c_times_q = sici(one_plus_c_times_q)

    prefact = 1 / (np.log(one_plus_c) - c / (one_plus_c))
    first_term = np.cos(q) * (Ci_one_plus_c_times_q - Ci_q)
    second_term = np.sin(q) * (Si_one_plus_c_times_q - Si_q)
    third_term = np.sin(c * q) / (q * one_plus_c)

    unfw = prefact * (first_term + second_term - third_term)

    return unfw


def _compute_exp(k, rs, tau_exp):
    """
    Returns Fourier Transform of the exp. decay profile.

    Args:
        tau : index of the exponential slope
        lambda_NFW : extending the NFW profile cut-off factor
    """

    denom = (1 + (tau_exp * rs * k) ** 2) ** 2
    uexp = 1 / denom

    return uexp


def compute_nfw_exp_mixed(k, c, rs, theta_prof):
    """
    Returns Fourier Transform of halo profile mimicing DESI ELGs.

    According to 2306.06319, the ELG halo profile is:
    p(r|M) = f_exp * exp(-r/(r_s * tau)) * (1-fexp)[NFW](r|M).

    Args:
        fexp : frac. of satellites following exp. profile
    """
    fexp, tau_exp, lambda_NFW = theta_prof

    unfw = compute_unfw(k, c, rs, lambda_NFW)
    uexp = _compute_exp(k, rs, tau_exp)

    u_tot = fexp * uexp + (1 - fexp) * unfw

    return u_tot
