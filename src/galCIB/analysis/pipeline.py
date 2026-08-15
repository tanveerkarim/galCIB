#from galCIB.galaxy import get_hod_model
from scipy.integrate import simpson
import numpy as np 

from .utils import bin_mat

class AnalysisModel:
    def __init__(self, survey, pk3d, bin_cl=False, limber_offset=0.5):
        """
        Initializes the model with fixed cosmology and survey properties.

        Args:
            cosmology: Cosmology object (with Mh, z, k, P_lin, etc.)
            survey: Survey object (with z, pz, compute_Wg, etc.)
            hod_name: Name of the default HOD model to use
            profile_type: e.g., "nfw" or "mixed"
        """
        
        self.survey = survey
        self.pk = pk3d
        self.cosmo = self.pk.cosmo
        self.bin_cl=bin_cl # if to bin Cl
        # Limber wavenumber k = (ell + limber_offset)/chi. 0.5 is the extended
        # Limber default; set 0.0 to reproduce DopplerCIB, which uses k = ell/chi.
        self.limber_offset = limber_offset
        
        self.geom_factor = self.cosmo.geom_factor
        self.Wg = survey.Wg
        self.Wcib = survey.Wcib
        self.Wmu = survey.Wmu 
        
        self.Nell = len(self.survey.ells)
        self.Nz = len(self.survey.z)
        self.Nnu = len(self.survey.nu_obs)
        self.Nnu_comb = self.Nnu * (self.Nnu + 1)//2
        
        self.cc_gI = np.array([self._get_color_correction(nu)for nu in self.survey.nu_obs])
        self.cc_II = np.array([
            self._get_color_correction(nu1) * self._get_color_correction(nu2)
            
            for i, nu1 in enumerate(self.survey.nu_obs)
            for j, nu2 in enumerate(self.survey.nu_obs)
            if j >= i
            ])
        
        self.cc_gI = self.cc_gI[:,None]
        self.cc_II = self.cc_II[:,None]
    
        # compute and cache mag bias auto-term
        self.Cmumu = self._cache_cl_mumu()
    
    def _kPk_interpolator(self, pk):
        """
        k and Pk need to be interpolated on the same grid 
        to calculate C_ell. The grid is:
        k = (ell+0.5)/chi(z)
        
        Returns:
            kz_grid : (Nell, Nz)
            Pk_grid : (Nell, Nz)
        """
        
        self.kz_grid = (self.survey.ells[:,np.newaxis]+self.limber_offset)/self.cosmo.chi[np.newaxis,:]
        
        # interpolate pk 
        
        pk_interp = np.zeros((self.Nell, self.Nz))
        
        for zidx in range(len(self.survey.z)):
            pk_interp[:,zidx] = np.interp(self.kz_grid[:,zidx],
                                          self.cosmo.k,
                                          pk[:,zidx])
            
        return pk_interp
        
    def compute_cl(self, pkz_xy, Wx, Wy):
        """
        Return C_ell gg using Limber. 
        
        Args:
            pkz_xy : interpolated P(k) of X and Y fields.
            Wx : Window of X
            Wy : Window of Y
        """
        
        integrand = self.geom_factor*Wx*Wy*pkz_xy
        cl = simpson(integrand,x=self.survey.z,axis=-1)
        
        return cl
    
    def _get_color_correction(self, f_obs):
        """
        Returns color correction multiplicative factor
        on CIB. 
        
        Each power of nu gets one cc_pl value
        """
        
        cc_pl = {}
        cc_pl[100] = 1.076
        cc_pl[143] = 1.017
        cc_pl[217] = 1.119
        cc_pl[353] = 1.097
        cc_pl[545] = 1.068
        cc_pl[857] = 0.995
        
        return cc_pl[f_obs]
    
    def _cache_cl_mumu(self):
        """
        Calculates the auto-power of mag bias. 
        
        Since this only depends on cosmology, needs to be
        only calculated once and can be cached. 
        """
        
        pmumu = self.pk.pk_mumu_2h
        
        # interpolate on the ell-to-k grid
        pmumu_int = self._kPk_interpolator(pmumu)
    
        # compute C_ell 
        Cmumu = self.compute_cl(pmumu_int, self.Wmu, self.Wmu)
        
        return Cmumu
    
    def update_cl(self, theta_cen=None, theta_sat=None,
                  theta_gal_prof=None, theta_cib_prof=None,
                  theta_sfr=None, theta_snu=None, theta_IR_hod=None,
                  theta_sn_gI=None, theta_sn_II = None,
                  hmalpha=1
                  ):
        """
        Recalculates C_ell based on parameters. 
        """
        
        pgg, pII, pgI, pgmu, pmuI = self.pk.compute_pk(theta_cen=theta_cen,
                                        theta_sat=theta_sat,
                                        theta_gal_prof=theta_gal_prof,
                                        theta_cib_prof=theta_cib_prof,
                                        theta_sfr=theta_sfr,
                                        theta_snu=theta_snu,
                                        theta_IR_hod=theta_IR_hod,
                                        hmalpha=hmalpha)
        
        # interpolate on the ell-to-k grid
        pgg_int = self._kPk_interpolator(pgg)
        pgmu_int = self._kPk_interpolator(pgmu)
        
        pgI_int = np.zeros((self.Nnu, self.Nell, self.Nz)) # Nnu, Nk, Nz
        pmuI_int = np.zeros((self.Nnu, self.Nell, self.Nz)) # Nnu, Nk, Nz
        for i in range(self.Nnu):
            pgI_int[i] = self._kPk_interpolator(pgI[i])
            pmuI_int[i] = self._kPk_interpolator(pmuI[i])
        
        pII_int = np.zeros((self.Nnu_comb, self.Nell, self.Nz))
        for i in range(self.Nnu_comb):
            pII_int[i] = self._kPk_interpolator(pII[i])
        
        # compute C_ell 
        Cgg = self.compute_cl(pgg_int, self.Wg, self.Wg)
        Cgmu = self.compute_cl(pgmu_int, self.Wg, self.Wmu)
        Cgg_tot = Cgg + 2*Cgmu + self.Cmumu
        
        CgI = self.compute_cl(pgI_int, self.Wg, self.Wcib)
        CmuI = self.compute_cl(pmuI_int, self.Wmu, self.Wcib)
        CgI_tot = CgI + CmuI
        
        CII = self.compute_cl(pII_int, self.Wcib, self.Wcib)
        
        # apply color correction
        CgI_tot= CgI_tot * self.cc_gI
        CII = CII * self.cc_II
        
        # apply shot-noise 
        CgI_tot = CgI_tot + theta_sn_gI[:,None]
        CII = CII + theta_sn_II[:,None]
        
        return Cgg_tot, CgI_tot, CII 