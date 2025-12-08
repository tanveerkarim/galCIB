"""
This module contains relevant information about
DESI Legacy Imaging Surveys galaxies.
"""

import numpy as np
import scipy.special as ss
from scipy.integrate import simpson
# import astropy.units as u

# import local modules
from .. import consts
# from .. import halo

# read in cosmological constants
OmegaM0 = consts.OmegaM0

# read in halo constants
Mh = consts.Mh_Msol
log10Mh = consts.log10Mh
dm = consts.log10Mh[1] - consts.log10Mh[0]

# read in survey information
z = consts.Plin["z"]


def Ncen_GHOD(log10Mc, sigmaM, Ac):
    """
    Returns num. of central galaxies per halo, as a function
    of halo mass.
    Based on Gaussian HOD Model (GHOD) Eq. 3.1 of 2306.06319

    Args:
        log10Mc : characteristic mass for a halo to host a central galaxy
        sigmaM : width of the distribution
        Ac : sets the size of the central galaxy sample
    Returns:
        res : A single number #FIXME: do we make log10Mc and sigmaM z dependent?
    """

    prefact = Ac / (np.sqrt(2 * np.pi) * sigmaM)
    exp_term = -0.5 / (sigmaM**2) * (log10Mh - log10Mc) ** 2
    res = prefact * np.exp(exp_term)

    return res


def z_evolution_model(params):
    """
    Returns redshift evolution model of HOD mass params.
    """

    mu_0, mu_p = params
    mu_X = mu_0 + mu_p * z / (1 + z)

    return mu_X


def Ncen(hod_params, gal_type):
    """
    Returns num. of central galaxies per halo, between 0 and 1
    as a function of halo mass.

    IR is based on 2.11 of 2310.10848.
    ELG is based on High Mass Quenched Model (mHMQ) Eq. 3.4 of 2306.06319

    Args:
        hod_params : HOD parameters to be constrained
            ELG hod_params: (gamma, log10Mc, sigmaM)
        gal_type : whether galaxy is ELG or IR
    Returns:
        res : vector of size (Mh,)
    """

    if gal_type == "ELG":
        # Functional form is:
        # <N_c(M)> = <N_c^(GHOD)(M)>[1+erf(gamma*(log10(Mh/Mc))/(sqrt(2)*sigmaM))]
        # From 3.4 of 2306.06319.

        gamma, log10Mc, sigmaM, Ac = hod_params
        erf_term = gamma * (log10Mh - log10Mc) / (np.sqrt(2) * sigmaM)
        second_term = 1 + ss.erf(erf_term)
        first_term = Ncen_GHOD(log10Mc, sigmaM, Ac)
        res = first_term * second_term

    elif gal_type == "IR":
        Mmin0, Mminp, IR_sigma_lnM = hod_params
        M_min_params = (Mmin0, Mminp)
        Mmin_z = z_evolution_model(M_min_params)
        # erf_term = np.log(Mh[:,np.newaxis]/10**Mmin_z[np.newaxis,:])/IR_sigma_lnM
        erf_term = (
            np.log(Mh[:, np.newaxis]) - Mmin_z[np.newaxis, :] * np.log(10)
        ) / IR_sigma_lnM

        res = 0.5 * (1 + ss.erf(erf_term))

    elif gal_type == "UNWISE":
        """
        Eqn 2.11 from 2310.10848
        
        N_c(M) = 0.5 * (1 + erf (ln(M/M_min)/sigma_lnM))
        """

        # M_min_params, sigma_lnM = hod_params
        mu_min0, mu_minp, sigma_lnM = hod_params
        M_min_params = (mu_min0, mu_minp)
        M_min_z = z_evolution_model(M_min_params)

        erf_term = (
            np.log(Mh[:, np.newaxis]) - M_min_z[np.newaxis, :] * np.log(10)
        ) / sigma_lnM
        res = 0.5 * (1 + ss.erf(erf_term))

    else:
        print("galaxy type not defined.")

    return res


def Nsat(hod_params, gal_type):
    """
    Returns num. of sat. gal. per halo

    Args:
        hod_params : HOD parameters satellite galaxies
            ELG:
                As : sets the size of the satellite galaxy sample
                M0 : cut-off halo mass from which satellites can be present
                M1 : Normalization factor
                alpha : controls the increase in satellite richness
                        with increasing halo mass
    Returns:
        res : vector of size (Mh,)
    """

    if gal_type == "ELG":
        As, M0, M1, alpha = hod_params
        M0 = 10**M0  # convert from log to regular space

        # flag for halo masses for which Mh - M0 > 0;
        # if Mh - M0 <= 0, then Nsat = 0. #FIXME: logic?
        flag = (Mh - M0) > 0
        res = np.zeros_like(Mh)
        res[flag] = As * ((Mh[flag] - M0) / M1) ** alpha

    elif gal_type == "UNWISE":
        """
        Eqn 2.11 from 2310.10848
        
        Nsat(M) = Nc(M) * Heaviside(M-M0) * ((M - M0)/M1)**alpha_s
        
        # note that M0 is fixed to 10^6 Msol in the UNWISE X CIB paper (private comm. with Ziang Yan)
        """

        # nc_hod_params, M0_params, M1_params, alpha_s = hod_params
        # Ncen components            # Nsat components
        mu_min0, mu_minp, sigma_lnM, mu_10, mu_1p, alpha_s = hod_params
        nc_hod_params = (mu_min0, mu_minp, sigma_lnM)
        M1_params = (mu_10, mu_1p)
        Nc_M = Ncen(nc_hod_params, gal_type=gal_type)

        # M0 = 10**z_evolution_model(M0_params)
        M1 = 10 ** z_evolution_model(M1_params)
        M0 = 1e6

        M_M0 = Mh - M0
        heaviside_term = np.heaviside(M_M0, 1)[:, np.newaxis]
        ratio_term = (M_M0[:, np.newaxis] / M1[np.newaxis, :]) ** alpha_s

        res = Nc_M * heaviside_term * ratio_term

    return res


def get_Wmu(dict_gal, mag_bias_alpha):
    """
    Returns magnification bias kernel as a func.
    of redshift of the galaxies.
    """

    pz = dict_gal["pz"]
    chi = consts.chi_list
    # Hz = consts.Hz_list

    # Eqn 6 of 1410.4502

    # prefactor = 3/2 * Omega_M0/c * H0^2/H(z) * (1+z) * chi(z) * (alpha - 1)
    # NOTE: if alpha == alpha(z), then that term should enter the integral
    # here we assume it is z-independent

    mag_bias_prefact = (
        3
        / 2
        * consts.OmegaM0
        / (consts.speed_of_light)
        * consts.H0**2
        / consts.Hz_list
        * (1 + consts.Plin["z"])
        * consts.chi_list
    ).decompose()
    mag_bias_prefact = mag_bias_prefact * (
        mag_bias_alpha - 1
    )  # assuming alpha is z-independent
    # mag_bias_prefact = (mag_bias_prefact * consts.H0**2/Hz * (1 + z) * chi).decompose() # to reduce to the same unit

    integrated_values = np.zeros_like(z)

    # integral of dz' * (1 - chi(z)/chi(z')) * dN/dz' from z'=z to z'=1090
    for i in range(len(z)):
        # consider the specific redshift from itself all the way to CMB;
        flag = z >= z[i]

        # dN/dz'
        pz_mag_bias = pz[flag]

        # chi(z')
        chi_list_mag_bias = chi[flag]

        # chi(z)
        chi_at_z = chi[i]

        ratio = 1 - chi_at_z / chi_list_mag_bias

        # integrand
        integrand = ratio * pz_mag_bias

        integrated_values[i] = simpson(integrand, x=z[flag])

        # return mag_bias_term

    # for i in range(len(z)): # loop over to get integrand values
    #   zspecific_indx = i

    #   # only consider bins above the specific index
    #   flag = (z >= z[zspecific_indx])
    #   ratio = chi[zspecific_indx]/chi[flag]

    #   # assuming constant alpha over z
    #   integrand_term = (1 - ratio) * (mag_bias_alpha - 1) * pz[flag]
    #   integrated_values[i] = simpson(y = integrand_term, x = z[flag])

    mag_bias_term = mag_bias_prefact * integrated_values

    return mag_bias_term  # shape (z,)


def get_Wgal(dict_gal):
    """
    Returns galaxy radial kernel as a func
    of redshift.
    """

    pz = dict_gal["pz"]

    return pz


def nbargal_halo(ncen, nsat, hmf):
    """
    Returns the expected galaxy count per z-bin based
    on the galaxy halo model.

    Args:
        ncen : number of centrals (Mh,)
        nsat : number of sat (Mh,)
        hmf : halo mass function (k, Mh, z)
    Returns:
    """

    Ntot = ncen + nsat  # total number of galaxies predicted by the halo model
    Ntot = Ntot[np.newaxis, :, np.newaxis]
    integrand = Ntot * hmf

    nbar = simpson(integrand, dx=dm, axis=1)

    return nbar


def galterm(params, _u, gal_type="ELG"):
    """
    Returns the second bracket in A13 of 2204.05299.
    Form is (Nc + Ns * u(k, Mh, z)).
    This corresponds to the galaxy term in calculating Pk.

    Note: For ELG we are only picking one effective redshift, hence
    no evolution of Ncen or Nsat.

    Args:
        params: MCMC parameters to be passed to HOD and 1-halo
                radial profile. The order of parameters are:
                gamma, log10Mc, sigmaM, Ac,
                As, M0, M1, alpha,
                fexp, tau, lambda_NFW.

    Returns:
        res : shape (k X Mh X z)
    """

    if gal_type == "ELG":
        params_Nc = params[:4]
        params_Ns = params[4:8]
    elif gal_type == "UNWISE":
        params_Nc = params[:3]
        params_Ns = params

    Nc = Ncen(params_Nc, gal_type=gal_type)  # (Mh,)
    Ns = Nsat(params_Ns, gal_type=gal_type)  # (Mh,)

    res = Nc[np.newaxis, :, :] + Ns[np.newaxis, :, :] * _u

    return res, Nc, Ns
