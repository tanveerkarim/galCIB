"""
Contains default callable HOD models. 
Includes DESI ELG mHMQ, GHOD and Zheng05 models.
"""

import numpy as np
from scipy.special import erf

from .utils import evolving_log_mass
from .registry import register_hod_model, is_model_registered

def Ncen_mHMQ(log10_Mh, theta):
    """
    Returns num. of central galaxies per halo, between 0 and 1
    as a function of halo mass. 
    
    ELG is based on High Mass Quenched Model (mHMQ) Eq. 3.4 of 2306.06319
    """
    
    gamma, log10Mc, sigmaM, Ac = theta
    
    erf_term = gamma * (log10_Mh - log10Mc)/(np.sqrt(2) * sigmaM)
    second_term = 1 + erf(erf_term)
    first_term = Ncen_GHOD(log10_Mh, (log10Mc, sigmaM, Ac))
    
    Ncen = first_term * second_term
    
    return Ncen

def Ncen_GHOD(log10_Mh, theta):
    """
    Returns num. of central galaxies per halo, as a function of halo mass. 
    Based on Gaussian HOD Model (GHOD) Eq. 3.1 of 2306.06319
    """
    
    log10Mc, sigmaM, Ac = theta
    
    exp_term = -0.5 * ((log10_Mh - log10Mc)/sigmaM)**2
    prefact = Ac/(np.sqrt(2 * np.pi) * sigmaM)
    
    Ncen = prefact * np.exp(exp_term)
    
    return Ncen

def Ncen_Z05(Mh, theta, z_over_1plusz=None):
    """
    Returns num. of central. galaxies (0 or 1) per halo.
    Based on Eqn 2.11 from 2310.10848
        
    N_c(M) = 0.5 * (1 + erf (ln(M/M_min)/sigma_lnM))
    """
    
    #mu0_Mmin, mup_Mmin, sigma_lnM = theta
    Mmin_z, sigma_lnM = theta
    # Mmin_z = 10**evolving_log_mass(mu0_Mmin, mup_Mmin, z_over_1plusz)
    Mmin_z = np.array([10**Mmin_z])
    
    erf_term = np.log(Mh[:,np.newaxis]/Mmin_z[np.newaxis,:])/sigma_lnM
    
    Ncen = 0.5 * (1 + erf(erf_term))
    
    return Ncen
    
def Nsat_ELG(Mh, theta):
    """
    Returns num. of sat. gal. per halo.
    Based on 3.5 of 2306.06319.
    """
    
    As, M0, M1, alpha_sat = theta

    # np.where evaluates BOTH branches, so ((Mh-M0)/M1)**alpha_sat used to be
    # computed for Mh <= M0 as well. With alpha_sat < 0 (the DESI-ELG fiducial
    # is -0.198) that gives 0**negative -> inf, and for non-integer alpha_sat a
    # negative base gives nan. The mask discarded them, but it raised
    # RuntimeWarnings and would surface as inf/nan under a sampler exploring M0.
    # Substitute a safe placeholder base before exponentiating.
    dM = Mh - M0
    base = np.where(dM > 0, dM, 1.0) / M1
    Nsat = np.where(dM > 0, As * base**alpha_sat, 0.0)

    return Nsat

def Nsat_Z05(Mh, theta, z_over_1plusz=None, **kwargs):
    
    """
    Returns num. of sat gal. per halo. 
    Eqn 2.11 from 2310.10848
    
    Nsat(M) = Nc(M) * Heaviside(M-M0) * ((M - M0)/M1)**alpha_s
    """
    
    ncen = kwargs.get("ncen", 1.0) # this implementation expects ncen
    M0, mu0_M1, mup_M1, alpha_sat = theta
    Mh_M0 = Mh - M0
    Mh_M0 = Mh_M0[:,np.newaxis]
    M1_z = 10**evolving_log_mass(mu0_M1, mup_M1, z_over_1plusz)
    M1_z = M1_z[np.newaxis,:]
    
    Nsat = np.heaviside(Mh_M0, 1)*(Mh_M0/M1_z)**alpha_sat
    
    if ncen is not None:
        Nsat *= ncen
    
    return Nsat 
        
# ===

# Store default params for each model here
_default_hod_params = {
    "Zheng05": dict(
        ncen_fn=Ncen_Z05,
        nsat_fn=Nsat_Z05,
        use_log10M_ncen=False,
        use_log10M_nsat=False,
        uses_z_ncen=True,
        uses_z_nsat=True,
    ),
    "DESI-ELG": dict(
        ncen_fn=Ncen_mHMQ,
        nsat_fn=Nsat_ELG,
        use_log10M_ncen=True,
        use_log10M_nsat=False,
        uses_z_ncen=False,
        uses_z_nsat=False,
    ),
}

for name, params in _default_hod_params.items():
    if not is_model_registered(name):
        register_hod_model(name=name, **params)
