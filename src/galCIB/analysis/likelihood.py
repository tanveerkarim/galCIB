# analysis/likelihood.py
"""
Gaussian likelihood for the gal x CIB + CIB x CIB data vector.

Parameter strategy (see THEORY_VALIDATION.md §6 and SYSTEMATICS.md):

  SAMPLED (9 physics + shot noise)
    SFR   : log10Mpeak, sigmaM0, tau, zc
    SED   : L0, beta_dust, T0, alpha_dust
    CIB HOD: log10Mmin_IR
    shot  : one per frequency for gI, one per frequency pair for II

  HELD FIXED, and why
    ELG HOD (8) + satellite profile (3)
        constrained externally from small-scale 3D clustering; gal x CIB alone
        cannot separate galaxy bias from CIB emissivity amplitude.
    eta_max = 1
        exactly degenerate with L0 (Yan+23: "the SFRD cannot be effectively
        constrained using our cross-correlation measurements alone ... eta_max
        is degenerate with the normalization").
    gamma_dust = 1.7
        MEASURED to be exactly degenerate with L0: varying it rescales the SED
        with a spread of 1.000000 across (nu, z), because the nu^-gamma branch
        is never reached (nu_rest = 635-2228 GHz vs nu0(z) = 3323-3590 GHz over
        0.8<z<1.6) and gamma enters only through the normalisation point nu0.
    sigma_lnM_IR = 0.4, alpha_s = 1
        fixed by Yan+23 following simulations.
    hmalpha = 1
        no 1h-2h smoothing, matching DopplerCIB.

NOTE on L0: the Y23 SED previously omitted the chi^2(1+z) of Eq. 2.43. After
that fix the natural scale of L0 changed by ~3.2e6 (old 5e-14 -> ~1.6e-7). The
default prior below is on log10(L0) to keep it scale-free.
"""

import numpy as np

# Parameters sampled, in vector order.
SAMPLED_PARAMS = [
    "log10Mpeak", "sigmaM0", "tau", "zc",          # SFR (M21)
    "log10L0", "beta_dust", "T0", "alpha_dust",    # SED (Y23)
    "log10Mmin_IR",                                # CIB HOD (Zheng05 central)
]

# Uniform prior boxes. SFR ranges follow Table 3 of 2310.10848; the SED ranges
# follow their Table 4. log10L0 is wide because the normalisation is arbitrary.
DEFAULT_PRIORS = {
    "log10Mpeak": (10.0, 14.0),
    "sigmaM0": (0.01, 4.0),
    "tau": (0.0, 1.0),
    "zc": (0.5, 3.0),
    "log10L0": (-12.0, 0.0),
    "beta_dust": (1.0, 3.0),
    "T0": (10.0, 40.0),
    "alpha_dust": (0.0, 1.0),
    "log10Mmin_IR": (10.0, 13.0),
    "log10_sn": (-4.0, 1.0),   # applied to every shot-noise amplitude
}

# Values held fixed (see module docstring).
FIXED = dict(eta_max=1.0, gamma_dust=1.7, sigma_lnM_IR=0.4, hmalpha=1.0)


class Sampler:
    """Gaussian likelihood over [CgI, CII] (optionally with Cgg).

    Args:
        cell_obj: AnalysisModel instance.
        data: 1D data vector, concatenated in the same order as `model_vector`.
        cov: covariance of `data`.
        theta_cen, theta_sat: ELG HOD, held fixed.
        theta_gal_prof, theta_cib_prof: satellite profiles, held fixed.
        n_nu: number of CIB frequencies.
        ell_mask: boolean mask (or index array) over the model's ell axis,
            selecting the bandpowers present in `data`. Required whenever the
            data uses a scale cut -- e.g. the l_min/l_max adopted in
            SYSTEMATICS.md -- while the model is evaluated on the full grid.
        include_cgg: prepend Cgg to the data vector. Off by default -- the ELG
            HOD is fixed externally, so Cgg adds no constraining power here and
            its shot noise is not modelled.
        priors: overrides for DEFAULT_PRIORS.
    """

    def __init__(self, cell_obj, data, cov,
                 theta_cen, theta_sat,
                 theta_gal_prof=None, theta_cib_prof=None,
                 n_nu=3, include_cgg=False, priors=None, ell_mask=None):
        self.cell_obj = cell_obj
        self.data = np.asarray(data)
        self.cov = np.asarray(cov)
        self.invcov = np.linalg.inv(self.cov)

        self.theta_cen = theta_cen
        self.theta_sat = theta_sat
        self.theta_gal_prof = theta_gal_prof
        self.theta_cib_prof = theta_cib_prof

        self.n_nu = n_nu
        self.n_nu_pairs = n_nu * (n_nu + 1) // 2
        self.include_cgg = include_cgg
        self.ell_mask = ell_mask

        self.priors = dict(DEFAULT_PRIORS)
        if priors:
            self.priors.update(priors)

        self.n_shot = self.n_nu + self.n_nu_pairs
        self.ndim = len(SAMPLED_PARAMS) + self.n_shot

        # constant term of the Gaussian log-likelihood; depends only on cov
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("covariance is not positive definite")
        self.log_ll_norm = -0.5 * logdet - 0.5 * len(self.data) * np.log(2 * np.pi)

    # -- parameter handling ------------------------------------------------

    def theta_parser(self, theta):
        """Split the sampling vector into the kwargs update_cl expects."""
        theta = np.asarray(theta, dtype=float)
        if theta.size != self.ndim:
            raise ValueError(f"expected {self.ndim} parameters, got {theta.size}")

        p = dict(zip(SAMPLED_PARAMS, theta[:len(SAMPLED_PARAMS)]))
        shot = theta[len(SAMPLED_PARAMS):]

        # theta_sfr for "M21": (eta_max, log10Mpeak, sigmaM0, tau, zc)
        theta_sfr = np.array([FIXED["eta_max"], p["log10Mpeak"],
                              p["sigmaM0"], p["tau"], p["zc"]])
        # theta_snu for "Y23": (L0, beta_dust, T0, alpha_dust, gamma_dust)
        theta_snu = np.array([10.0 ** p["log10L0"], p["beta_dust"],
                              p["T0"], p["alpha_dust"], FIXED["gamma_dust"]])
        # theta_IR_hod for Zheng05 Ncen: (log10Mmin, sigma_lnM)
        theta_IR = np.array([p["log10Mmin_IR"], FIXED["sigma_lnM_IR"]])

        theta_sn_gI = 10.0 ** shot[:self.n_nu]
        theta_sn_II = 10.0 ** shot[self.n_nu:]
        return theta_sfr, theta_snu, theta_IR, theta_sn_gI, theta_sn_II

    def model_vector(self, theta):
        """Theory prediction, concatenated to match `data`."""
        t_sfr, t_snu, t_IR, sn_gI, sn_II = self.theta_parser(theta)

        cgg, cgI, cII = self.cell_obj.update_cl(
            theta_cen=self.theta_cen,
            theta_sat=self.theta_sat,
            theta_gal_prof=self.theta_gal_prof,
            theta_cib_prof=self.theta_cib_prof,
            theta_sfr=t_sfr,
            theta_snu=t_snu,
            theta_IR_hod=t_IR,
            theta_sn_gI=sn_gI,
            theta_sn_II=sn_II,
            hmalpha=FIXED["hmalpha"],
        )
        def _sel(a):
            a = np.atleast_2d(a)
            if self.ell_mask is not None:
                a = a[:, self.ell_mask]
            return a.ravel()

        parts = [_sel(cgI), _sel(cII)]
        if self.include_cgg:
            parts.insert(0, _sel(np.atleast_1d(cgg)))
        model = np.concatenate(parts)
        if model.size != self.data.size:
            raise ValueError(
                f"model has {model.size} elements but data has "
                f"{self.data.size}; check ell_mask / include_cgg")
        return model

    # -- likelihood --------------------------------------------------------

    def log_prior(self, theta):
        """Uniform (hard-box) prior. Returns 0.0 inside, -inf outside."""
        theta = np.asarray(theta, dtype=float)
        if theta.size != self.ndim:
            return -np.inf
        if not np.all(np.isfinite(theta)):
            return -np.inf

        for name, val in zip(SAMPLED_PARAMS, theta[:len(SAMPLED_PARAMS)]):
            lo, hi = self.priors[name]
            if not (lo <= val <= hi):
                return -np.inf

        lo, hi = self.priors["log10_sn"]
        if not np.all((theta[len(SAMPLED_PARAMS):] >= lo)
                      & (theta[len(SAMPLED_PARAMS):] <= hi)):
            return -np.inf
        return 0.0

    def loglike(self, theta):
        """Gaussian log-likelihood."""
        try:
            model = self.model_vector(theta)
        except (ValueError, FloatingPointError):
            return -np.inf
        if not np.all(np.isfinite(model)):
            return -np.inf

        r = model - self.data
        return -0.5 * r @ self.invcov @ r + self.log_ll_norm

    def logpost(self, theta):
        """Log-posterior; -inf outside the prior without evaluating the model."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

    def prior_bounds(self):
        """(lo, hi) arrays in sampling order, for samplers that want a box."""
        lo = [self.priors[n][0] for n in SAMPLED_PARAMS]
        hi = [self.priors[n][1] for n in SAMPLED_PARAMS]
        slo, shi = self.priors["log10_sn"]
        lo += [slo] * self.n_shot
        hi += [shi] * self.n_shot
        return np.array(lo), np.array(hi)
