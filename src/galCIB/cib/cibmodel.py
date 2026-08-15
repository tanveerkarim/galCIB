"""
This sets up the CIB model class that can take user-defined SFR and Snu
models. It has the M21 and Y23 models implemented as defaults. 
"""

import numpy as np 
from scipy.integrate import simpson 
# from .sfrmodel import SFRModel
# from .snumodel import SnuModel
# from .registry import _lazy_register_defaults
from .utils import _compute_BAR_grid

class CIBModel:
    """
    Container class for CIB emissivity model.
    
    Parameters
    ----------
    cosmo : Cosmology
        Cosmology object providing Mh, z, chi, etc.
    hod_IR : HODModel
        HOD model for IR-emitting galaxies (not necessarily same as clustering sample).
    sfr_model : callable
        SFRModel instance.
    snu_model : callable
        SnuModel instance.
    subhalo_mf : callable, optional
        Custom subhalo mass function: fn(m_over_M) → dN/dlog10m.
    Nm_sub : int
        Number of subhalo mass bins to use.
    """
    
    def __init__(self, hod_IR, sfr_model, snu_model,
                 subhalo_mf=None, Nm_sub=98):
        
        self.hod_IR = hod_IR
        self.sfr_model = sfr_model
        self.snu_model = snu_model
        self.survey = self.snu_model.survey
        self.cosmo = self.snu_model.cosmo 
        self.subhalo_mf = subhalo_mf 
        
        self.NMh = len(self.cosmo.Mh)
        self.Nz = len(self.cosmo.z)
        
        # Geometric prefactor
        self.geom_prefact = self.cosmo.chi**2 * (1 + self.cosmo.z)
        KC = 1e-10 # Kennicutt Constant
        self.geom_prefact_over_KC = (self.geom_prefact / KC)[None,None,:] # (1, 1, Nz)
        
        # Precompute subhalo mass grid and mass function
        self.fsub = self.sfr_model.fsub 
        
        self.Mhc = self.cosmo.Mh * (1-self.fsub)
        self.log10_Mhc = np.log10(self.Mhc)
        
        log10_m_min = 5
        self.m_sub_grid = np.logspace(log10_m_min, 
                                self.log10_Mhc,
                                Nm_sub)  # (Nm_sub, NMh)
        
        self.dlog10m = np.log10(self.m_sub_grid[1, :]/self.m_sub_grid[0, :])  # shape: (NMh,)
        self.m_over_Mhc = (self.m_sub_grid /self.Mhc[None, :])[...,None]  # (Nm_sub, NMh,1)
        
        # Compute SHMF grid
        self.subhalo_mf_grid = self._compute_subhalo_mf()[...,None]
        
        # Compute subhalo BAR grid 
        self.subhalo_BAR_grid = _compute_BAR_grid(cosmo=self.cosmo,Mh=None, 
                                                  m=self.m_sub_grid) #(Nm, NMh)
        
    def _compute_subhalo_mf(self):
        
        # NOTE: m_over_M must be bound outside the `if`, otherwise passing a
        # custom subhalo_mf raises UnboundLocalError.
        m_over_M = self.m_sub_grid/self.cosmo.Mh

        if self.subhalo_mf is None:
            """
            Default based on 10 of 0909.1325.
            """
            self.subhalo_mf = lambda m_over_M : 0.3 * m_over_M**-0.7 * np.exp(-9.9 * m_over_M**2.5) * np.log(10)

        return self.subhalo_mf(m_over_M)
        
            
    def update(self, theta_sfr, theta_snu, theta_hod_IR):
        """
        Update internal cache given current proposal.
        
        fraction of the mass of the halo that is in form of
        sub-halos. We have to take this into account while calculating the
        star formation rate of the central halos. It should be calulated by
        accounting for this fraction of the subhalo mass in the halo mass
        central halo mass in this case is (1-f_sub)*mh where mh is the total
        mass of the halo.
        for a given halo mass, f_sub is calculated by taking the first moment
        of the sub-halo mf and and integrating it over all the subhalo masses
        and dividing it by the total halo mass.
        """
        
        self._sfr = (self.sfr_model(theta_sfr))#[None,:,:]             # (1, Nm, Nz)
        
        # Compute effective SED
        self._snu = self.snu_model(theta_snu)  # shape (Nnu_fine, Nz)
        
        # Compute IR galaxies 
        self._Ncen_IR = (self.hod_IR.ncen(theta_hod_IR))[None,:,:]  # (1, Nm, Nz)
        
        self._dj_central = self._compute_djc(self._sfr, self._snu, self._Ncen_IR) # (Nnu, Nm, Nz)
        self._dj_sub = self._compute_djsub(self._sfr, self._snu, theta_sfr) # (Nnu, Nm, Nz)
            
    def _compute_djc(self, sfr, snu, Ncen_IR):
        """
        Compute emissivity from central galaxies: (Nnu, Nm, Nz)
        
        Based on A6 of 2204.05299
        
        djc/dlog Mh (Mh, z) = chi^2 (1+z) * SFRc/K * S_nu(z)
        """
        
        # Ncen_IR = 1 #FIXME: for test 
        #print(f'set Ncen_IR = 1 for testing.')
        SFRc = sfr * Ncen_IR # (1, Nm, Nz)
        djc = self.geom_prefact_over_KC * SFRc     # (1, Nm, Nz)
        return snu[:,None,:] * djc            # (Nnu, Nm, Nz)

    def _compute_djsub(self, sfr, snu, theta_sfr):
        """
        Compute emissivity from subhalos: (Nnu, Nm, Nz)
        
        Based on A7 of 2204.05299
        djsub/dlogMh (Mh,z) = chi^2 (1+z) * S_nu(z) * int (dN/dlog m_sub) (m_sub | Mh) * SFRsub/K * dlogm_sub
        
        Based on 2.41 of 2310.10848
        SFRsub = min (SFR(msub), msub/Mh * SFR(Mh))
        """
        
        #m_sub = self.m_sub_grid      # (Nm_sub, Nm, 1)
        sfr_M = sfr[None,...]                # (1, NMh, Nz)

        # Calculate SFR(m) 
        sfrII = self.sfr_model.evaluate_from_BAR(self.subhalo_BAR_grid, self.m_sub_grid,
                                                 theta_sfr) # (Nmsub, NMh,1)
        
        # Choose the minimum: min(SFR(m), m/Mh * SFR(M))
        m_over_M = self.m_over_Mhc          # (Nm_sub, NMh, 1)
        sfrI = m_over_M * sfr_M                    # (Nm_sub, NMh, Nz)
        
        sfr_sub = np.minimum(sfrII, sfrI) # (Nm_sub, NMh, Nz)

        # Subhalo mass function
        dNdlog10m = self.subhalo_mf_grid  # (Nm_sub, NMh, 1)
        
        integrand = sfr_sub * dNdlog10m  # (Nm_sub, NMh, Nz)
        sfr_sub_total = np.empty((self.NMh, self.Nz))
        
        for i in range(self.NMh):
            sfr_sub_total[i, :] = simpson(integrand[:, i, :],
                                  dx=self.dlog10m[i],
                                  #x=np.log10(self.m_sub_grid[:,i]),
                                  axis=0) # (NMh, Nz)

        djsub = self.geom_prefact_over_KC * sfr_sub_total  # (Nm, Nz)
        return snu[:,None,:] * djsub                  # (Nnu, Nm, Nz)

    def get_djc(self):
        return self._dj_central

    def get_djsub(self):
        return self._dj_sub

