# cib/default_snu.py
"""
Contains default S_nu models. 
M21 : Non-parametric model from Planck.
Y23 : Parametric model inspired by S12.
"""

import numpy as np 
from astropy.io import fits

from scipy.special import lambertw
from scipy.interpolate import CubicSpline

from scipy.constants import h, k, c  # Planck and Boltzmann in SI
kB_over_h = k/h # for nu0z 
h_over_kB = h/k 

MPC_TO_M = 3.085677581e22  # meters
WATT_TO_JY = 1e26          # Jy
L_SUN = 3.828e26           # watts

###--PARAMETRIC MODEL BASED ON Y23--###

def snu_Y23_factory(nu_prime, z):
    """
    Factory returns a callable: S_nu(theta_SED, nu_prime)
    
    Args:
        nu_prime : nu' = nu*(1+z) grid over which to evaluate SED
    """

    planck_prefactor = 2*h*nu_prime**3/c**2

    def snu_Y23(theta_SED):
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
    
        L0, beta_dust, T0, alpha_dust, gamma_dust = theta_SED
        Td = _compute_Tdust(T0, alpha_dust, z)
        nu0z = _compute_nu0_z(beta_dust, Td, gamma_dust)
        
        # calculate SED 
        flag = nu_prime < nu0z[np.newaxis, :]
        
                
        theta = np.where(flag, 
                        _compute_prenu0(beta_dust, Td, nu_prime, planck_prefactor), 
                        _compute_postnu0(nu_prime, gamma_dust))

        # normalize SED such that theta(nu0) = 1
        theta_normed = np.where(flag, 
            theta/_compute_prenu0(beta_dust, Td, nu0z, 2*h*nu0z**3/c**2),
            theta/_compute_postnu0(nu0z, gamma_dust))
                
        return theta_normed*L0
    
    return snu_Y23

def _compute_prenu0(beta, Td, nu, planck_prefactor):
    """
    Returns the gray-body part of the SED.
    """
    
    res = nu**beta * _compute_B_nu(nu, Td, planck_prefactor)
    return res 

def _compute_postnu0(nu, gamma_dust):
    """
    Returns the exponential decay part of the SED.
    """
    
    res = nu**(-gamma_dust)
    return res

def _compute_Tdust(T0, alpha_dust, z):
    """
    Returns dust temperature as a func. of z.
    """
    return T0 * (1 + z)**alpha_dust

def _compute_nu0_z(beta, Td, gamma):
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
    
    K = (gamma + beta + 3)
    lambert_term = lambertw(-K*np.exp(-K))
    full_term = K + lambert_term
    nu0z = np.real(kB_over_h * Td * full_term )
    
    return nu0z

def _compute_B_nu(nu, Td, prefact):
    """
    Returns Planck's blackbody function.
    
    Args:
        nu : frequency array in Hz of shape (nu, z)
        Td : dust temperature  of shape (z,)
        
    Returns:
        res : of shape (nu, z)
    """
    # Pre-factor depending only on nu, shape (nu, z)
    #prefact = hp_times_2_over_c2 * nu**3

    # Ensure Td is broadcast correctly along the redshift dimension
    Td_re = Td[np.newaxis, :]  # Shape (1, z)
    
    # Calculate the exponential term
    x = h_over_kB * nu/Td_re # Shape (nu, z)

    # Compute the Planck function
    res = prefact / np.expm1(x)  # Shape (nu, z)
    
    return res

###--NON-PARAMETRIC SED MODEL BASED ON M21--###    
def snu_M21_factory(selected_freqs, cosmo, fdata="../data/filtered_snu_planck.fits"):
    """
    Factory function that returns a callable S_nu(theta, z)
    using precomputed non-parametric SEDs from Planck (M21).
    
    Returns:
        snu_M21(theta_snu) callable
    """
    
    hdulist = fits.open(fdata)
    redshifts = hdulist[1].data
    snu_eff = hdulist[0].data[:-1, :]  # in Jy/Lsun  # -1 because we are
    # not considering the 3000 GHz channel which comes from IRAS
    hdulist.close()
    
    def select_rows_by_frequency(snu, selected_freqs):
        """
        Only return user-specified frequencies.
        """
        
        planck_freqs = np.array([100., 143., 217., 353., 545., 857.])  # shape (6,)
        
        selected_freqs = np.asarray(selected_freqs)
        freq_to_index = {freq: i for i, freq in enumerate(planck_freqs)}
        try:
            indices = [freq_to_index[f] for f in selected_freqs]
        except KeyError as e:
            raise ValueError(f"Requested frequency {e.args[0]} not in the list of available frequencies.")
        return snu[indices]
    
    snu_subselect = select_rows_by_frequency(snu_eff, selected_freqs)
    
    interpolator = CubicSpline(redshifts,
                               snu_subselect,
                               axis=1,
                               extrapolate=False)
    
    def snu_M21(theta_unused):
        """
        Evaluates the SED model for given nu (in GHz) and z.
        
        Args:
            theta_unused: placeholder for interface compatibility
            nu : frequency array (Hz)
            z  : redshift array
        Returns:
            SED of shape (len(nu), len(z))
        """
        
        return interpolator(cosmo.z)

    return snu_M21

_SNU_FACTORIES = {
    "Y23": snu_Y23_factory,
    "M21": snu_M21_factory,
    }

def get_snu_factory(name):
    return _SNU_FACTORIES[name]
