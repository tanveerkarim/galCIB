# cib/default_models.py

"""
Contains the default SFR model based on 2006.16329
"""

import numpy as np
from galCIB.galaxy.utils import evolving_log_mass

def sfr_default(BAR_grid, z_ratio):
    """
    Returns the default SFR model 
    """
    
    def eta_fn(M, z, theta_eta, **kwargs):
        """
        Returns star formation efficiency parameter.
        
        From 2.38 of 2310.10848.
        eta (M, z) = eta_max * exp(-0.5((ln M - ln Mpeak(z))/sigmaM(z))^2)
        M = Mh and z = z so these are fixed. 
        sigmaM represents range of halo masses over which star formation is efficient. 
        """
        
        z_ratio = kwargs.get('z_ratio', None)
        eta_max, log10Mpeak, sigmaM0, tau, zc = theta_eta
        
        # Mpeak evolving as a func. of z 
        #Mpeak_z = 10**evolving_log_mass(mu0_peak, mup_peak, z_ratio)
        Mpeak_z = 10**log10Mpeak
        # 2.39 of 2310.10848
        sigmaM_z = sigmaM0 - tau * np.maximum(0, zc - z)              # (Nz,)
        
        # sigmaM_z = np.where(M < Mpeak_z, sigmaM0, 
        #                 sigmaM0*(1-tau/zc * np.maximum(0, zc-z)))  
        # sigmaM_z = np.where(M_b < Mpeak_z_b, sigmaM0,
        #                     sigmaM0-tau*np.maximum(0,zc-z))
        
        
        Nz = z_ratio.shape[0]
        sigmaM_z = sigmaM0 - tau * np.maximum(0, zc - z)              # (Nz,)
        
        M_b = np.broadcast_to(M[..., None], M.shape + (Nz,))          # (..., Nz)
        
        Mpeak_b = np.broadcast_to(Mpeak_z, M_b.shape)                 # (..., Nz)
        sigmaM_b = np.where(M_b < Mpeak_b, sigmaM0,
                            np.broadcast_to(sigmaM_z, M_b.shape))     # (..., Nz)

        expterm = -0.5 * ((np.log(M_b) - np.log(Mpeak_b)) / sigmaM_b) ** 2
            
        
        # sigmaM_eff = np.where(M_b < Mpeak_z_b, sigmaM0, sigmaM_z_b)
        # expterm = -0.5 * ((np.log(M_b) - np.log(Mpeak_z_b))/sigmaM_eff)**2
        
        eta_z = eta_max * np.exp(expterm)
        
        return eta_z
    
    def sfr_model(M, z, theta_eta):
        eta_vals = eta_fn(M,z,theta_eta,
                          z_ratio=z_ratio)

        if eta_vals.shape != BAR_grid.shape:
            raise ValueError(f"Shape mismatch: eta_vals {eta_vals.shape}, BAR_grid {BAR_grid.shape}")
        
        return eta_vals * BAR_grid
    
    return sfr_model
