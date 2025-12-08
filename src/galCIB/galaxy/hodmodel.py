"""
This sets up the galaxy HOD class that calculates relevant properties such as
HOD models. It has a few default HOD models and can also take user-defined models.
"""


class HODModel:
    """
    Container class for a Halo Occupation Distribution (HOD) model.

    Attributes
    ----------
    name : str
        Unique identifier for the model.
    ncen_fn : callable
        Function that returns ⟨N_cen(M, z)⟩ given M, z, and theta.
    nsat_fn : callable
        Function that returns ⟨N_sat(M, z)⟩ given M, z, and theta.
    z : array
        Redshift array
    use_log10M : bool
        True -> Function takes log10Mh, False -> Function takes Mh
    """

    def __init__(
        self,
        name,
        ncen_fn,
        nsat_fn,
        cosmo,
        use_log10M_ncen=False,
        use_log10M_nsat=False,
        uses_z_ncen=False,
        uses_z_nsat=False,
    ):
        self.name = name
        self._ncen_fn = ncen_fn
        self._nsat_fn = nsat_fn
        self.cosmo = cosmo
        self.z = self.cosmo.z
        self.z_ratio = self.z / (1 + self.z)
        self.use_log10M_ncen = use_log10M_ncen
        self.use_log10M_nsat = use_log10M_nsat
        self.uses_z_ncen = uses_z_ncen
        self.uses_z_nsat = uses_z_nsat

    def ncen(self, theta_cen):
        M_input = self.cosmo.log10Mh if self.use_log10M_ncen else self.cosmo.Mh

        if self.uses_z_ncen:
            return self._ncen_fn(M_input, theta_cen, self.z_ratio)
        else:
            return self._ncen_fn(M_input, theta_cen)

    def nsat(self, theta_sat, **kwargs):
        M_input = self.cosmo.log10Mh if self.use_log10M_nsat else self.cosmo.Mh

        if self.uses_z_nsat:
            return self._nsat_fn(M_input, theta_sat, self.z_ratio, **kwargs)
        else:
            return self._nsat_fn(M_input, theta_sat, **kwargs)

    # def evaluate(self, theta_cen, theta_sat, log10Mh, z=None):
    #     """Evaluate both N_cen and N_sat for given halo mass
    #     and redshift grids."""
    #     ncen_vals = self.ncen(log10Mh, theta_cen, z)
    #     nsat_vals = self.nsat(log10Mh, theta_sat, z, ncen=ncen_vals)
    #     return {
    #         "N_cen": ncen_vals,
    #         "N_sat": nsat_vals
    #     }
