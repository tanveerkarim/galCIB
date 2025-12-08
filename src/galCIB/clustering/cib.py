"""
This module contains functions useful for CIB halo modelling.
"""

import numpy as np
from .. import consts

# integrates using simpson method
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Lambert W function solver
from scipy.special import lambertw

# physical constants
c_light = consts.speed_of_light
c_light_kms = c_light / 1000
KC = consts.KC
L_sun = consts.L_sun
hp = consts.hp
kB = consts.k_B
hp_over_kB = consts.hp_over_kB
hp_times_2_over_c2 = consts.hp_times_2_over_c2

# cosmology constants
Om0 = consts.OmegaM0
Ode0 = consts.Ode0
bar_c = consts.bar_c
bar_sub = consts.bar_sub

# halo constants
Mh = consts.Mh_Msol
Mhc = consts.Mhc_Msol  # central galaxies halo mass based on fsub
ms = consts.ms_Msol  # subhalo mass grid based on fsub
ms_to_Mhc = consts.ms_to_Mhc_Msol  # subhalo mass grid as a fraction with Mhc
# ms_to_Mh = consts.ms_to_Mh_Msol # subhalo mass grid as a fraction with Mhc
hmfz = consts.hmfz  # halo mass function
subhalomf = consts.subhalomf  # subhalo mass function
log10ms = np.expand_dims(np.log10(ms), axis=-1)

# survey constants
z = consts.Plin["z"]
chi = consts.chi_list

# galaxy constants
# dict_gal = consts.dict_gal['ELG']
# chi = dict_gal['chi'] # comoving distance
# z = dict_gal['z']

# SED constants
dgamma = 1.7
nu_primes = consts.nu_primes
planck_nu_list = consts.nu0
ghz = consts.ghz
z_cib_planck = consts.redshifts_M23
chi_cib = consts.chi_cib

###--RADIAL KERNEL---###


def get_W_cib(z_cib):
    """
    Returns redshift kernel of CIB field.
    """
    a = 1 / (1 + z_cib)

    return a


###--END OF RADIAL KERNEL--###


def cibterm(params, u, cib_model):
    """
    Returns the first bracket in A13 of 2204.05299.
    This corresponds to the CIB term in calculating Pk.

    Args:
        params : SFR and SED parameters
        unfw : Fourier transform the NFW profile inside the halo. (k, Mh, z)
        cib_model : Name of the CIB model to be tested

    Returns:
        res : shape (k, nu, Mh, z)
    """

    if cib_model == "Y23":
        L0 = params[0]  # overall normalization parameter of the SED
        params_sfr = params[1:-3]
        params_seff = params[-3:]  # last three params
        seff = L0 * Seff(params_seff, model=cib_model)  # shape (nu, z)
    elif cib_model == "M21":
        params_sfr = params
        seff = Seff(params=None, model=cib_model)  # Non-parametric model
    else:
        print("Did not input correct model.")

    djc = djc_dlogMh(params_sfr, seff, cib_model)
    djsub = djsub_dlogMh(params_sfr, seff, cib_model)
    final_term = (
        djc[:, np.newaxis, :, :] + djsub[:, np.newaxis, :, :] * u[np.newaxis, :, :, :]
    )

    return final_term, djc, djsub


###--START OF C_ell HELPER FUNCTIONS--###
def djc_dlogMh(params_sfr, seff, model):
    """
    Returns the emissivity of central galaxies per log halo mass.

    from A6 of 2204.05299
    djc_dlogMh (Mh, z) = chi^2 * (1+z) * SFRc/K * S^eff_nu (z)

    Args:
        params : SFR and Seff model parameters, depends on model
            order is [beta, T0, alpha]
        model : 'S12', 'M21' or 'Y23'
    Returns:
        jc : matrix of shape (nu, Mh, z)
    """

    # fraction of the mass of the halo that is in form of
    # sub-halos. We have to take this into account while calculating the
    # star formation rate of the central halos. It should be calculated by
    # accounting for this fraction of the subhalo mass in the halo mass
    # central halo mass in this case is (1-f_sub)*mh where mh is the total
    # mass of the halo.
    # for a given halo mass, f_sub is calculated by taking the first moment
    # of the sub-halo mf and and integrating it over all the subhalo masses
    # and dividing it by the total halo mass.

    prefact = chi**2 * (1 + z) / KC  # shape (z,)
    sfrc = SFRc(params_sfr, model)  # shape (Mh, z)
    # broadcast properly for multiplication
    sfrc_re = sfrc[np.newaxis, :, :]  # shape (1, Mh, z)
    seff_re = seff[:, np.newaxis, :]  # shape (nu, 1, z)
    prefact_re = prefact[np.newaxis, np.newaxis, :]  # shape (1, 1, z)

    jc = prefact_re * sfrc_re * seff_re

    return jc


def djsub_dlogMh(params_sfr, seff, model):
    """
    Returns the emissivity of satellite galaxies per log halo mass.

    from A7 of 2204.05299
    djsub_dlogMh (Mh, z) = chi^2 * (1+z) S^eff_nu (z) * int (dN_dlogm_sub (m_sub|Mh) * SFRsub/K * dlogm_sub)

    Args:
        nu : measurement frequency
        z : redshift
        model : 'S12', 'M21' or 'Y23'
    Returns:
        jc : matrix of shape (nu, Mh, z)
    """

    prefact = chi**2 * (1 + z)
    prefact = prefact / KC  # shape (z,)

    # integral in A7 of 2204.05299
    sfrsub = SFRsub(params_sfr, model)
    integrand = sfrsub * np.expand_dims(subhalomf, axis=-1)
    integral = simpson(
        y=integrand, x=log10ms, axis=0
    )  # integrate along the ms axis, shape (Mh, z)

    # broadcast shapes properly for multiplication
    prefact_re = prefact[np.newaxis, np.newaxis, :]  # shape (1,1,z)
    integral_re = integral[np.newaxis, :, :]  # shape (1,Mh,z)
    seff_re = seff[:, np.newaxis, :]

    # jsub of shape (nu, Mh, z)
    jsub = prefact_re * integral_re * seff_re

    return jsub


###--END OF C_ell HELPER FUNCTIONS--###

###--START OF S_eff MODELING--###


def Tdust(T0, alpha):
    """
    Returns dust temp as a function of z.
    From 1309.0382.

    Returns:
        res: of shape (z,)
    """

    res = T0 * (1.0 + z_cib_planck) ** alpha
    return res


def B_nu(nu, Td):
    """
    Returns Planck's blackbody function.

    Args:
        nu : frequency array in Hz of shape (nu, z)
        Td : dust temperature  of shape (z,)

    Returns:
        res : of shape (nu, z)
    """
    # Pre-factor depending only on nu, shape (nu, z)
    prefact = hp_times_2_over_c2 * nu**3

    # Ensure Td is broadcast correctly along the redshift dimension
    Td_re = Td[np.newaxis, :]  # Shape (1, z)

    # Calculate the exponential term
    x = hp_over_kB * nu / Td_re  # Shape (nu, z)

    # Compute the Planck function
    res = prefact / (np.exp(x) - 1)  # Shape (nu, z)

    return res


def nu0_z(beta, Td):
    """
    Returns the pivot frequency as a function of redshift.

    For modified blackboy approximation,
    we have dln[v^beta * B_nu(Td)]/dlnnu = -gamma
    for nu=nu0 from 2.27 of 2310.10848.

    In order to find nu0 which is redshift dependent, we need to
    use the Lambert W function. of the form x = a + be^(c*x).
    Solution is given by: x = a - W(-bce^(ac))/c, check
    https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.special.lambertw.html
    for reference.

    Here, x = nu0, a = K/h_KT, b = -a, c = -h_KT,
    where K = (gamma + beta + 3) and h_KT = h/(KT)

    Args:
        beta: controls "grayness" of blackbody function
        Td: dust temperature as a function of z
    Returns:
        nu0z: pivot frequencies as a function of redshift. Shape (z,)
    """

    h_kT = hp_over_kB / Td  # shape (z,)
    K = dgamma + beta + 3

    a = K / h_kT
    b = -1 * a
    c = -1 * h_kT

    x = a - lambertw(-b * c * np.exp(a * c)) / c
    nu0z = np.real(x)  # only take the real value

    return nu0z


def Theta(params):
    """
    Returns the modified SED for gray-body function
    normalized to 1 at pivot freq.

    From 2.27 of 2310.10848.

    theta(nu') = nu'^beta * B_nu' (T) for n < nu0
               = nu'^(-gamma) for n >= nu0

    Args:
        params: (beta, T0, alpha)
            beta: controls "grayness" of blackbody function
            T0, alpha: dust parameters
    Returns:
        theta_normed : of shape (nu, z)
    """

    beta, T0, alpha = params
    theta = np.zeros_like(nu_primes)  # shape (nu, z)
    Td = Tdust(T0, alpha)
    nu0z = nu0_z(beta, Td)

    def prenu0(beta, Td, nu):
        """
        Returns the gray-body part of the SED.
        """

        res = nu**beta * B_nu(nu, Td)

        return res

    def postnu0(gamma, nu):
        """
        Returns the exponential decay part of the SED.
        """

        res = nu ** (-gamma)

        return res

    # calculate SED
    flag = nu_primes < nu0z[np.newaxis, :]

    theta = np.where(
        flag, prenu0(beta, Td, nu_primes).value, postnu0(dgamma, nu_primes)
    )

    # normalize SED such that theta(nu0) = 1
    theta_normed = np.where(
        flag,
        theta / prenu0(beta, Td, nu0z[np.newaxis, :]).value,
        theta / postnu0(dgamma, nu0z[np.newaxis, :]),
    )

    return theta_normed


def convolve_with_Planck(seff):
    """
    Returns convolved seff with Planck filters.
    """

    # load relevant filter arrays
    # filtarray = {}

    # store Planck filter convolved SED values per filter and per z
    seff_convolved = np.empty((len(planck_nu_list), len(z_cib_planck)))

    def interp_slice(nu_grid_slice, S_eff_slice, filtfreq):
        """
        Returns interpolated values of along Planck filter curves per slice.
        This function is needed to vectorize the operation across multiple
        redshifts simultaneously.

        Args:
            nu_grid_slice : Along z direction, nu values
            S_eff_slice : Along z direction, S_eff values
            filtfreq : Planck filter frequency bins
        Returns:
            res : interpolated S_eff values for filter # ff_idx
        """

        interp_func = interp1d(
            nu_grid_slice, S_eff_slice, bounds_error=False, fill_value=0
        )

        res = interp_func(filtfreq)
        return res

    # slice_interpolator = lambda z_idx, filtfreq: interp_slice(nu_primes[:,z_idx[0]],
    #                                            seff[:, z_idx[0]], filtfreq)
    slice_interpolator = None

    planck_nu_names = (planck_nu_list / ghz).astype(int)
    for i in range(len(planck_nu_names)):
        # read in filter curves
        filter_str_name = str(planck_nu_names[i])
        fname = f"data/filters/HFI__avg_{filter_str_name}_CMB_noise_avg_Apod5_Sfull_v302_HNETnorm.dat"
        filtarray = np.loadtxt(fname, usecols=(1, 2))

        filter_response = filtarray[:, 1]  # response curve
        filtfreq = filtarray[:, 0] * ghz  # convert to Hz
        area = simpson(y=filter_response, x=filtfreq)  # area under filter
        filter_response = filter_response / area  # normalize by area

        # perform interpolation
        arr_zidx = np.arange(consts.nu_primes.shape[1]).reshape(-1, 1)
        seff_interpolated = np.apply_along_axis(
            slice_interpolator, axis=1, arr=arr_zidx, filtfreq=filtfreq
        )

        # weight raw SED by filter response curve
        tnu_seff = seff_interpolated * filter_response[np.newaxis, :]

        # integrate total flux along filter curve
        seff_convolved[i] = simpson(y=tnu_seff, x=filtfreq, axis=1)

    # print(seff_interpolated)

    return seff_interpolated, seff_convolved


def Seff(params, model):
    """
    Returns the effective spectral energy distribution (SED)
    which is the fraction of IR radiation at the
    rest-frame frequency (1 + z)nu.

    It is defined as the mean flux density per total solar luminosity.

    Args:
        params: (beta, T0, alpha)
            beta: controls "grayness" of blackbody function
            T0, alpha: dust parameters
    Returns:
        seff : of shape (nu, z)
    """

    if model == "Y23":
        seff = Theta(params)
        seff_int, seff = convolve_with_Planck(seff)  # model images per Planck filter

        # divide out by distance
        seff = seff / (1 + z_cib_planck[np.newaxis, :])
        seff = seff / chi_cib[np.newaxis, :] ** 2

    elif model == "M21":
        seff = consts.snu_eff_z
    else:
        print("Not correct model.")

    # return seff_int, seff
    return seff


###--END OF S_eff MODELING--###


###--START OF SFR MODELING--###
def SFR(etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, is_sub=False):
    """
    Returns star formation rate for models
    M21 and Y23.

    Args:
        is_sub : flag for whether SFR is for subhaloes
    """

    eta_val = eta(etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, is_sub=is_sub)

    if is_sub:
        sfr = eta_val * bar_sub
    else:
        sfr = eta_val * bar_c

    return sfr


def SFRc(params, model):
    """
    Returns star formation rate of central galaxies as a function of halo parameters and model.

    Args:
        params : model parameters
        Mhc : halo mass of central galaxies
        model : model name
    """

    from . import gal  # for HOD of IR galaxies

    if model == "M21":
        # SFR_c (Mh, z) = eta (Mh, z) * BAR (Mh, z)

        (
            etamax,
            mu_peak0,
            mu_peakp,
            sigma_M0,
            tau,
            zc,
            Mmin_IR0,
            Mmin_IRp,
            IR_sigma_lnM,
        ) = params
        IR_hod_params = (Mmin_IR0, Mmin_IRp, IR_sigma_lnM)
        sfr = SFR(etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc)

        mean_N_IR_c = gal.Ncen(
            IR_hod_params, gal_type="IR"
        )  # [:,np.newaxis] #FIXME: only if no z evolution model of N_c

        # mean_N_IR_c = 1 #FIXME: only to match M21

        test = False
        if test:
            mean_N_IR_c = 1
            print("you set meanIR to 1 by hand for testing.")

        sfrc = sfr * mean_N_IR_c

    elif model == "Y23":
        mu_peak0, mu_peakp, sigma_M0, tau, zc, Mmin_IR, IR_sigma_lnM = params
        IR_hod_params = (Mmin_IR, IR_sigma_lnM)
        # Model cannot constrain etamax so set to 1.
        # Normalization is absorbed by L0 param in SED.
        sfr = SFR(
            etamax=1,
            mu_peak0=mu_peak0,
            mu_peakp=mu_peak0,
            sigma_M0=sigma_M0,
            tau=tau,
            zc=zc,
        )
        mean_N_IR_c = gal.Ncen(IR_hod_params, gal_type="IR")
        sfrc = sfr * mean_N_IR_c

    else:
        print("Not correct model.")

    return sfrc  # , mean_N_IR_c


def SFRsub(params, model):
    """
    Returns SFR of subhalos.
    """

    if model == "M21":
        # from 2.41 of 2310.10848
        # SFRs (m|M) = min(SFR(m), m/M * SFR(M))

        etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, _, _, _ = params

        option1 = SFR(etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, is_sub=True)

        option2 = (
            ms_to_Mhc[:, :, np.newaxis]
            * SFR(etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, is_sub=False)[
                np.newaxis, :, :
            ]
        )  # proper broadcasting to get shape (ms, Mh, z)

        sfrs = np.minimum(option1, option2)

        return sfrs

    elif model == "Y23":
        mu_peak0, mu_peakp, sigma_M0, tau, zc, _, _ = params

        option1 = SFR(
            etamax=1,
            mu_peak0=mu_peak0,
            mu_peakp=mu_peakp,
            sigma_M0=sigma_M0,
            tau=tau,
            zc=zc,
            is_sub=True,
        )

        option2 = (
            ms_to_Mhc[:, :, np.newaxis]
            * SFR(
                etamax=1,
                mu_peak0=mu_peak0,
                mu_peakp=mu_peakp,
                sigma_M0=sigma_M0,
                tau=tau,
                zc=zc,
                is_sub=False,
            )[np.newaxis, :, :]
        )  # proper broadcasting to get shape (ms, Mh, z)

        sfrs = np.minimum(option1, option2)

        return sfrs


# DONE
def eta(etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, is_sub):
    """
    Returns star formation efficiency parameter.

    From 2.38 of 2310.10848.
    eta (M, z) = eta_max * exp(-0.5((ln M - ln Mpeak(z))/sigmaM(z))^2)
    M = Mh and z = z so these are fixed.
    sigmaM represents range of halo masses over which star formation is efficient.

    Args:
        etamax : highest star formation efficiency
        mu_peak0 : peak of halo mass contributing to IR emissivity at z = 0
        mu_peakp : rate of change of halo mass contributing to IR emissity at higher z
        sigma_M0 : halo mass range contributing to IR emissivity below peak mass
        zc : the redshift below which the mass window for star formation starts to evolve
        tau : rate of zc evolution
        is_sub : flag for whether this applies to subhaloes

    Returns:
        eta_val : of shape (Mh, z)
    """

    # Reshape M and z for broadcasting
    if is_sub:
        M_re = np.expand_dims(ms, axis=-1)  # Shape (ms, Mh, 1)
    else:
        M_re = np.expand_dims(Mhc, axis=-1)  # Shape (Mh, 1)
    z_re = z[np.newaxis, :]  # Shape (1, len(z))

    # Mpeak = mu_peak0 + mu_peakp * z_re/(1+z_re) # M_peak may change with z
    Mpeak = mu_peak0 + mu_peakp * z / (1 + z)
    Mpeak = 10**Mpeak
    # parametrization based on 2.39 of 2310.10848.
    sigmaM = np.where(
        M_re < Mpeak, sigma_M0, sigma_M0 - tau * np.maximum(0, zc - z_re)
    )  # Shape (len(M), len(z))
    expnumterm = (np.log(M_re) - np.log(Mpeak)) ** 2
    expterm = np.exp(-expnumterm / (2 * sigmaM**2))
    eta_val = etamax * expterm

    return eta_val


###--END OF SFR MODELING--###
