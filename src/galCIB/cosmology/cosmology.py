"""
This module defines the fixed cosmology used in the analysis.
"""


import numpy as np

from colossus.cosmology import cosmology as cc
from colossus.lss import mass_function
from astropy import units as u 
import astropy.cosmology.units as cu 

class Cosmology:
    def __init__(self, zs, ks, Mh, 
                 colossus_cosmo_name = 'planck18',
                 use_little_h = False):
        
        """
        Returns Cosmology Class with user-defined choices.
        
        Args:
            zs : redshift range
            ks : k range 
            Mh : Halo mass range
            colossus_cosmo_name : str referring to built-in colossus cosmology
            use_little_h : bool, True if using little h convention
        """
        
        self.cosmo_name = colossus_cosmo_name
        self.z = zs
        self.k = ks
        self.Mh = Mh
        self.log10Mh = np.log10(Mh)
        self.use_little_h = use_little_h
        
        self.pk_grid, self.dlnpk_dlnk, self.hmf_grid, self.cosmo = self._load_colossus()
        
        # initialize frequently used quantities
        self._cache_quantities()
        
        # project k,M,z on to mesh
        self._generate_mesh_grid()
        
        # cache values on the grid
        self.log10Mh_grid = np.log10(self.Mh_grid)
        
        # calculate geometric factor for C_ell calculation
        #self.geom_factor_c_ell = 1/self.c * self.Hz/self.chi**2
        self.geom_factor = self.dchi_dz/self.chi**2
        
    def _generate_mesh_grid(self,use_mesh_grid=True):
        """
        Returns k,M,z mesh for efficient calculations
        downstream.
        """
        
        self.k_grid, self.Mh_grid, self.z_grid = np.meshgrid(self.k, self.Mh, self.z,
                                              indexing='ij')
        
    def _load_colossus(self):
        """
        Returns P_mm (k), dln[Pk]/dln[k],
        Halo Mass Function and Halo Bias
        from colossus. 
        """
        
        cosmo = cc.setCosmology(self.cosmo_name)
        
        pk_grid = np.zeros((len(self.k),len(self.z)))
        hmf_grid = np.zeros((len(self.Mh),len(self.z)))
        
        if self.use_little_h is False:
            k_range = self.k / cosmo.h # Units of h/Mpc for colossus
            Mh_range = self.Mh * cosmo.h # Msol/h for colossus
        else: 
            k_range = self.k # units of h/Mpc 
            Mh_range = self.Mh # units of Msol/h
        
        for i in range(len(self.z)):
            tmp_pk = cosmo.matterPowerSpectrum(k_range, 
                                               self.z[i],
                                               model = 'camb')
            
            # Convert dn/dlnM to dn/dlog_10M, using ln(10) factor. 
            tmp_hmf = mass_function.massFunction(x=Mh_range, z=self.z[i],
                                                     mdef='200m', model='tinker08',
                                                     q_in='M', q_out='dndlnM') * np.log(10)
            
            # whether to convert HMF into h-less unit
            if self.use_little_h is False:
                pk_grid[:,i] = tmp_pk/cosmo.h**3 
                hmf_grid[:,i] = tmp_hmf*cosmo.h**3
            else:
                pk_grid[:,i] = tmp_pk
                hmf_grid[:,i] = tmp_hmf
        
        tmp_dlnpk_dlnk = cosmo.matterPowerSpectrum(k_range,
                                                    self.z[0], # same in all z
                                                    model = 'camb',
                                                    derivative = True)
        
        return pk_grid, tmp_dlnpk_dlnk, hmf_grid, cosmo
    
    def _cache_quantities(self):
        """
        Caches most frequently used quantities downstream. 
        """
        
        # comoving distance in units of Mpc
        self.chi = self._calculate_chi()
            
        # H(z) 
        self.Hz = self.cosmo.Hz(self.z) # units of km/s/Mpc
        
        # H0 
        self.H0 = self.cosmo.H0 # units of km/s/Mpc
        
        # Omega_M(z = 0)
        self.Om0 = self.cosmo.Om0 
        
        # Omega_Lambda(z = 0)
        self.Ode0 = self.cosmo.Ode0
        
        # Omega_b
        self.Ob_z = self.cosmo.Ob(self.z)
        
        # Omega_M
        self.Om_z = self.cosmo.Om(self.z)
        
        # rho_crit 
        self.rho_crit = self.cosmo.rho_c(self.z)*(u.Msun * cu.littleh**2/u.kpc**3) # units of Msol * h^2/kpc^3
        self.rho_crit = self.rho_crit.to(u.Msun/u.Mpc**3, 
                                         cu.with_H0(self.H0*u.km/u.s/u.Mpc)).value # convert to units of Msol/Mpc^3
        
        # mean_density of matter at (z=0)
        self.mean_density0 = self.cosmo.rho_c(0)*(u.Msun * cu.littleh**2/u.kpc**3) # units of Msol * h^2/kpc^3
        self.mean_density0 = self.mean_density0.to(u.Msun/u.Mpc**3, 
                                                   cu.with_H0(self.H0*u.km/u.s/u.Mpc)).value # convert to units of Msol/Mpc^3
        self.mean_density0 = self.Om0*self.mean_density0
        
        # speed of light in km/s 
        self.c = 299792.458  # km/s
        
        # dchi/dz 
        self.dchi_dz = self.c/self.cosmo.Hz(self.z)
         
    def _calculate_chi(self):
        
        chi = np.zeros_like(self.z)
        for i in range(len(self.z)):
            chi[i] = self.cosmo.comovingDistance(0,
                                                 self.z[i])
            
        if self.use_little_h is False:
            chi = chi/self.cosmo.h 
            
        return chi 
             
    def get_pk_grid(self):
        """
        Returns the precomputed P(k,z) grid.
        """   
        
        return self.pk_grid
    
    def get_hmf_grid(self):
        """
        Returns precomputed HMF(z,Mh)
        """
        
        return self.hmf_grid
    
    def get_k_range(self, use_little_h):
        """k on the requested convention. self.k follows self.use_little_h;
        internally colossus is fed k/h and M*h (see _load_colossus)."""
        if use_little_h:
            res = self.k / self.cosmo.h   # 1/Mpc -> h/Mpc
        else:
            res = self.k

        return res

    def get_Mh_range(self, use_little_h):
        """Halo mass on the requested convention (see get_k_range)."""
        if use_little_h:
            res = self.Mh * self.cosmo.h  # Msun -> Msun/h
        else:
            res = self.Mh

        return res
    
    def save_grids(self, filename):
        np.savez(filename, 
                    pk=self.pk_grid, 
                    z=self.z, k=self.k,
                    hmf=self.hmf_grid,
                    Mh=self.Mh)
        
    @classmethod
    def load_grids(cls, filename):
        data = np.load(filename)
        instance = cls.__new__(cls)  # create instance without __init__
        instance.pk_grid = data['pk']
        instance.z = data['z']
        instance.k = data['k']
        instance.hmf = data['hmf']
        instance.Mh = data['Mh']
        
        return instance