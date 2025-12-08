"""
Contains 1-halo and 2-halo term P(k) functions.
"""

from scipy.integrate import simpson
from .utils import (
    ensure_nm_nz_shape,
    compute_Pgg_1h,
    compute_Pgg_2h,
    compute_PgI_1h,
    compute_PgI_2h,
    compute_PII_1h,
    compute_PII_2h,
    compute_Puv_tot,
    compute_Pmumu_2h,
)


class PkBuilder:
    def __init__(
        self,
        hod_model,
        cib_model,
        gal_prof_model,
        cib_prof_model,
        theta_cen=None,
        theta_sat=None,
        theta_gal_prof=None,
        theta_cib_prof=None,
        theta_sfr=None,
        theta_snu=None,
        theta_IR_hod=None,
    ):
        self.hod = hod_model
        self.cib = cib_model
        self.cosmo = self.cib.cosmo
        self.gal_prof_model = gal_prof_model
        self.cib_prof_model = cib_prof_model
        self.gal_u = self.gal_prof_model.u
        self.cib_u = self.cib_prof_model.u
        self.k = self.cosmo.k
        self.z = self.cosmo.z
        self.log10Mh = self.cosmo.log10Mh
        self.dlog10Mh = self.log10Mh[1] - self.log10Mh[0]  # equal spacing
        # FIXME: let user pass unequal spacing as an option
        self.hmf = self.cosmo.hmf_grid
        self.hmfxbias = (
            self.hmf * self.gal_prof_model.bnu
        )  # FIXME: bias model choice, do we care if galaxy or CIB?

        self.theta_cen = theta_cen
        self.theta_sat = theta_sat
        self.theta_gal_prof = theta_gal_prof
        self.theta_cib_prof = theta_cib_prof
        self.theta_sfr = theta_sfr
        self.theta_snu = theta_snu
        self.theta_IR_hod = theta_IR_hod

        # cache the P_mumu term
        self._cache_mag_bias_auto_term()

        ## Whether to update u profile based on profile types ##
        # We determine which profiles are mixed
        is_gal_mixed = self.gal_prof_model.profile_type != "nfw"
        is_cib_mixed = self.cib_prof_model.profile_type != "nfw"

        # Case 1: Both are mixed
        if is_gal_mixed and is_cib_mixed:
            print("INFO: Both GALAXY and CIB profiles are mixed.")
            self._cache_u_profile = self._cache_all_profiles

        # Case 2: Only Galaxy is mixed
        elif is_gal_mixed and not is_cib_mixed:
            print("INFO: GALAXY profile is mixed; CIB profile is static ('nfw').")
            self._cache_u_profile = self._cache_galaxy_only

        # Case 3: Only CIB is mixed
        elif not is_gal_mixed and is_cib_mixed:
            print("INFO: CIB profile is mixed; GALAXY profile is static ('nfw').")
            self._cache_u_profile = self._cache_cib_only

        # Case 4: Both are static
        else:
            print("INFO: Both GALAXY and CIB profiles are static ('nfw').")
            self._cache_u_profile = self._cache_nothing

    def _cache_galaxy_integral(self):
        """
        Cache [Ncen(M,z) + Nsat(M,z) * u(k,M,z)]
        Galaxy term from A11 of 2204.05299
        """

        gal_u = self.gal_u

        self.ncen = ensure_nm_nz_shape(
            self.hod.ncen(self.theta_cen), len(self.cosmo.Mh), len(self.cosmo.z)
        )  # shape (Nm, Nz)
        self.nsat = ensure_nm_nz_shape(
            self.hod.nsat(self.theta_sat), len(self.cosmo.Mh), len(self.cosmo.z)
        )  # shape (Nm, Nz)

        self.nsat_u = self.nsat * gal_u  # useful in multiple places in Pk
        self.ncen_plus_nsat_u = self.ncen + self.nsat_u

        # pre-compute 2h mass integral term for speedup
        integrand = self.hmfxbias * self.ncen_plus_nsat_u  # (Nk, NMh, Nz)
        self.Ig = simpson(integrand, dx=self.dlog10Mh, axis=1)  # (Nk, Nz)

    def _cache_cib_integral(self):
        """
        Cache [djc(nu,M,z) + djs(nu,M,z) * u(k,M,z)]
        CIB term from A9 of 2204.05299
        """

        cib_u = self.cib_u

        # update CIB model
        self.cib.update(self.theta_sfr, self.theta_snu, self.theta_IR_hod)

        self.djc = self.cib.get_djc()[:, None, :, :]  # (Nnu, 1, NMh, Nz)
        self.djsub = self.cib.get_djsub()[:, None, :, :]

        self.djsub_u = (
            self.djsub * cib_u[None, :, :, :]
        )  # useful in multiple places in Pk
        self.djc_plus_djsub_u = self.djc + self.djsub_u

        # pre-compute 2h mass integral term for speedup
        integrand = self.djc_plus_djsub_u * self.hmfxbias  # (Nnu, Nk, NMh, Nz)
        self.Icib = simpson(integrand, dx=self.dlog10Mh, axis=2)  # (Nnu, Nk, Nz)

    def _cache_mag_bias_auto_term(self):
        """
        If magnification bias alpha != 1, P_mumu needs to
        be calculated and added to the galaxy analysis.

        Since cosmology is fixed, this needs to be only
        computed once.
        """

        self.pk_mumu_2h = compute_Pmumu_2h(self)

    ##-- Configurations of u profile caching --##
    # Note: if either galaxy or CIB profile is NFW, then no need to
    # update u(k,M,z) when theta_prof changes.

    def _cache_nothing(self):
        """Helper: Both profiles are static. Do nothing."""
        pass

    def _cache_galaxy_only(self):
        """Helper: Updates GALAXY profile, leaves CIB untouched."""
        self.gal_prof_model.update_theta(self.theta_gal_prof)
        self.gal_u = self.gal_prof_model.u

    def _cache_cib_only(self):
        """Helper: Updates CIB profile, leaves GALAXY untouched."""
        self.cib_prof_model.update_theta(self.theta_cib_prof)
        self.cib_u = self.cib_prof_model.u

    def _cache_all_profiles(self):
        """Helper: Updates both GALAXY and CIB profiles."""
        self.gal_prof_model.update_theta(self.theta_gal_prof)
        self.gal_u = self.gal_prof_model.u

        self.cib_prof_model.update_theta(self.theta_cib_prof)
        self.cib_u = self.cib_prof_model.u

    # def _cache_u_profile(self):
    #     """
    #     Cache new satellite radial profile
    #     """

    #     # self.prof_model.update_theta(self.theta_prof)
    #     # self.u = self.prof_model.u

    #     # if prof_model.profile_type is 'nfw' then do not update
    #     if self.gal_prof_model.profile_type != 'nfw':
    #         self.gal_prof_model.update_theta(self.theta_gal_prof)
    #         self.gal_u = self.gal_prof_model.u

    #     if self.cib_prof_model.profile_type != 'nfw':
    #         self.cib_prof_model.update_theta(self.theta_cib_prof)
    #         self.cib_u = self.cib_prof_model.u

    def _compute_nbar(self):
        # A12 of 2204.05299

        self.nbar = simpson(
            self.hmf * (self.ncen + self.nsat), dx=self.dlog10Mh, axis=0
        )
        self.nbar2 = self.nbar**2  # useful for multiple Pk

    def _update_theta(
        self,
        theta_cen,
        theta_sat,
        theta_gal_prof,
        theta_cib_prof,
        theta_sfr,
        theta_snu,
        theta_IR_hod,
    ):
        """
        Update model parameters and recompute the Pk.
        """

        # new theta
        self.theta_cen = theta_cen
        self.theta_sat = theta_sat
        self.theta_gal_prof = theta_gal_prof
        self.theta_cib_prof = theta_cib_prof
        self.theta_sfr = theta_sfr
        self.theta_snu = theta_snu
        self.theta_IR_hod = theta_IR_hod

        # update cache
        self._cache_u_profile()
        self._cache_galaxy_integral()
        self._cache_cib_integral()

        # nbar
        self._compute_nbar()

    def compute_pk(
        self,
        theta_cen=None,
        theta_sat=None,
        theta_gal_prof=None,
        theta_cib_prof=None,
        theta_sfr=None,
        theta_snu=None,
        theta_IR_hod=None,
        hmalpha=1,
        return_full_matrix_II=False,
    ):
        """
        Return Pgg, PII, PgI.

        Args:
            theta_cen, theta_sat = galaxy parameters
            theta_gal_prof = sat. galaxy radial profile parameters
            theta_cib_prof = sat. CIB radial profile parameters
            theta_sfr = SFR parameters
            theta_snu = Snu parameters
            theta_IR_hod = IR galaxy Ncen parameters
            hmalpha = 1h to 2h transition relaxation parameter
            return_full_matrix_II = Whether to return full nu x nu' matrix or only unique one (upper-triangle)

        """

        # update relevant cached values.
        self._update_theta(
            theta_cen,
            theta_sat,
            theta_gal_prof,
            theta_cib_prof,
            theta_sfr,
            theta_snu,
            theta_IR_hod,
        )

        self.pk_gg_2h, self.pk_gmu_2h = compute_Pgg_2h(self)
        self.pk_gg_1h = compute_Pgg_1h(self)

        if return_full_matrix_II is False:
            self.pk_II_2h, self.twoh_pairs = compute_PII_2h(
                self, return_full_matrix=return_full_matrix_II
            )
            self.pk_II_1h, self.oneh_pairs = compute_PII_1h(
                self, return_full_matrix=return_full_matrix_II
            )
        else:
            self.pk_II_2h = compute_PII_2h(
                self, return_full_matrix=return_full_matrix_II
            )
            self.pk_II_1h = compute_PII_1h(
                self, return_full_matrix=return_full_matrix_II
            )

        self.pk_gI_2h, self.pk_muI_2h = compute_PgI_2h(self)
        self.pk_gI_1h = compute_PgI_1h(self)

        self.pk_gg_tot = compute_Puv_tot(self.pk_gg_1h, self.pk_gg_2h, hmalpha)
        self.pk_II_tot = compute_Puv_tot(self.pk_II_1h, self.pk_II_2h, hmalpha)
        self.pk_gI_tot = compute_Puv_tot(self.pk_gI_1h, self.pk_gI_2h, hmalpha)

        return (
            self.pk_gg_tot,
            self.pk_II_tot,
            self.pk_gI_tot,
            self.pk_gmu_2h,
            self.pk_muI_2h,
        )

    def get_theta(self):
        """
        Print parameters.
        """

        print(f"central = {self.theta_cen}")
        print(f"sat = {self.theta_sat}")
        print(f"gal profile = {self.theta_gal_prof}")
        print(f"cib profile = {self.theta_cib_prof}")
