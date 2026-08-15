#cib/snumodel.py
import numpy as np
from scipy.integrate import simpson
from .registry import get_snu_model

class SnuModel:
    def __init__(self, 
                 name, 
                 cosmo,
                 survey,
                 nu_prime=None, 
                 m21_fdata="../data/filtered_snu_planck.fits",
                 ):
        
        self.name = name
        self.survey = survey
        self.cosmo = cosmo 
        self.z = self.cosmo.z
        if nu_prime is not None:
            self.nu_prime = nu_prime
            # nu_prime = nu_obs*(1+z); recover the observed grid it came from.
            self.nu_obs_grid = np.asarray(nu_prime) / (1 + self.z) \
                if np.ndim(nu_prime) == 2 else np.asarray(nu_prime)
        else:
            self.nu_prime, self.nu_obs_grid = self._generate_nu_prime_grid()
        self.model_fn = self._build_model(name, m21_fdata)

    def _build_model(self, name, data_dir):
        factory = get_snu_model(name)
        if name == "Y23":
            return factory(self.nu_prime, self.z)
        elif name == "M21":
            return factory(self.nu_prime, self.cosmo, data_dir)
        else:
            raise ValueError(f"Unknown snu model: {name}")

    def __call__(self, theta_snu):
        if self.name == 'Y23':
            snu_unfilt = self.model_fn(theta_snu)
            # The SED is evaluated at nu_prime = nu_obs*(1+z), so row i of
            # snu_unfilt corresponds to OBSERVED frequency nu_obs[i]. Planck
            # bandpasses are defined in the observed frame, so the integration
            # abscissa must be nu_obs -- using nu_prime applies (1+z) the wrong
            # way and returns the SED at nu_filt/(1+z).
            nu_obs = np.asarray(self.nu_obs_grid)
            if nu_obs.ndim == 1:
                nu_obs = np.tile(nu_obs, (len(self.z), 1))   # (Nz, Nwv)
            else:
                nu_obs = nu_obs.T
            self.snu_eff = self.apply_filter_to_sed(sed=snu_unfilt.T,
                                                    freq_sed=nu_obs)
            # Eq. 2.43 of 2310.10848:
            #   S_eff[(1+z)nu, z] ∝ Theta[(1+z)nu, z] / (chi^2 (1+z))
            # CIBModel._compute_djc multiplies by chi^2(1+z)/K, so without this
            # the Y23 emissivity carries a spurious chi^2(1+z) in z. The legacy
            # DopplerCIB code divided it out explicitly (clustering/cib.py::Seff).
            self.snu_eff = self.snu_eff / (
                self.cosmo.chi**2 * (1 + self.z))[None, :]
            
        elif self.name == 'M21':
            self.snu_eff = self.model_fn(theta_snu)
            
        return self.snu_eff
        
        #return snu_unfilt

    def _generate_nu_prime_grid(self):
        """
        Returns nu' = nu*(1+z) grid in units of Hz. 
        
        Useful to pass to any parametric SED model that 
        calculates SED as a function of nu. 
        """
        
        ghz = 1e9
        nu = np.linspace(1e2, 4e3, 10000) * ghz          # observed frame, Hz
        return nu[:, None] * (1 + self.z)[None, :], nu   # (Nnu, Nz), (Nnu,)
    
    
    def apply_filter_to_sed(self, sed, freq_sed):
        """
        Returns predicted flux of a given SED through a single filter.

        Args:
            sed : (Nz, Nwv) array of SEDs (Nz samples, Nwv frequencies)
            freq_sed : (Nwv,) array of frequencies for the SED
            filter_key : key to select filter from self.filters dict

        Returns:
            flux : (Nz,) flux for each SED through selected filter
        """
        
        filter_key = self.survey.nu_obs
        flux_effective = np.zeros((len(filter_key),
                                   len(self.z))) # Nnu, Nz
        
        ii = 0
        for fkey in filter_key:
            filt_freq, filt_response = self.survey.filters[fkey]  # unpack filter arrays
            
            sed = np.atleast_2d(sed)  # ensure shape (Nz, Nwv)
            norm = simpson(filt_response, x=filt_freq)

            # Integrate each SED over the filter response with interpolation
            flux = np.array([simpson(np.interp(filt_freq, w_row, 
                                        s_row, left=0.0, right=0.0) * filt_response, 
                                x=filt_freq) for w_row, s_row in zip(freq_sed, sed)])
            
            flux_effective[ii] = flux/norm
            ii +=1 

        return flux_effective
