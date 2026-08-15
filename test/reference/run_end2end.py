"""
End-to-end consistency: measured DESI DR2 ELG x Lenz CIB -> galCIB theory ->
Gaussian likelihood.

Exercises every ingredient together: the measured bandpowers and their
NaMaster/iNKA covariance (SYSTEMATICS.md), the halo model with the Y23 SED
fixes (THEORY_VALIDATION.md §4.2-4.3), and the rewritten Sampler.

The CIB amplitude L0 and the shot noise are pure linear parameters, so for each
(bin, frequency) they are solved in closed form and the residual chi^2 then
tests the model SHAPE against the data -- which is the part the halo model
actually predicts.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from galCIB import (AnalysisModel, CIBModel, PkBuilder, SatProfile,  # noqa: E402
                    SFRModel, SnuModel, Survey, get_hod_model)
from galCIB.analysis.likelihood import SAMPLED_PARAMS, Sampler  # noqa: E402
from galCIB.utils.io import load_my_filters  # noqa: E402

REPO = H.REPO
MEAS = REPO / "data" / "measurements" / "gal_cib"
NU = [353, 545, 857]
ZBINS = ["0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6"]
LMIN, LMAX_USE = 100.0, 600.0

meas = np.load(MEAS / "cl_galxcib_4.0e+20_gp40_dz0.2_lmax2000_lmask4000_nran2.npz")
snr = np.load(MEAS / "snr_dz0.2_lmax2000_lmask4000_nran2.npz")
leff = meas["leff"]
use = (leff > LMIN) & (leff < LMAX_USE)
print(f"[data] {len(ZBINS)} bins x {len(NU)} freqs, {use.sum()} bandpowers "
      f"in {LMIN:.0f}<l<{LMAX_USE:.0f}")

# n(z) per tomographic bin, straight from the measurement's own catalog cache
cat = np.load(MEAS / "cache" / "cat_data.npz")
zcat, wcat = cat["z"], cat["w"]

zgrid = np.linspace(0.01, 4.0, 200)
kgrid = np.logspace(-4, 1, 500)
filters = load_my_filters(str(REPO / "data" / "filters"), nu_obs=NU)

# Y23 fiducial (Table 3/4 of 2310.10848). L0 is solved for, so its value here
# only sets the scale of the linear fit.
FID = dict(log10Mpeak=11.79, sigmaM0=2.48, tau=0.5, zc=2.15,
           log10L0=-7.0, beta_dust=1.98, T0=21.13, alpha_dust=0.21,
           log10Mmin_IR=11.47)

print(f"\n{'bin':>9} {'nu':>4} {'A_fit':>10} {'chi2/dof':>9} {'dof':>4} "
      f"{'data S/N':>9}")
results = {}
for zk in ZBINS:
    lo, hi = (float(x) for x in zk.split("-"))
    sel = (zcat >= lo) & (zcat < hi)
    cnt, edges = np.histogram(zcat[sel], bins=40, range=(lo, hi), weights=wcat[sel])
    ctr = 0.5 * (edges[1:] + edges[:-1])
    pz = np.interp(zgrid, ctr, cnt.astype(float), left=0.0, right=0.0)
    pz = pz / simpson(pz, x=zgrid)   # compute_Wg does not normalise

    survey = Survey(z=zgrid, pz=pz, mag_alpha=2.225, cib_filters=filters,
                    ells=leff, nside=2048)
    cos = H.build_cosmology(zgrid, kgrid, pk_ref=None,
                            mass=10 ** np.arange(9, 15.05, 0.1))
    survey.compute_windows(cos)

    prof = SatProfile(cos, profile_type="nfw")
    hod_IR = get_hod_model("Zheng05", cos)
    sfr = SFRModel(name="M21", hod=hod_IR, fsub=H.FSUB)
    snu = SnuModel(name="Y23", cosmo=cos, survey=survey)
    cib = CIBModel(hod_IR=hod_IR, sfr_model=sfr, snu_model=snu)
    pk = PkBuilder(hod_model=get_hod_model("DESI-ELG", cos), cib_model=cib,
                   gal_prof_model=prof, cib_prof_model=prof)
    ana = AnalysisModel(survey=survey, pk3d=pk)

    _, cgI, _ = ana.update_cl(
        theta_cen=H.THETA_CEN, theta_sat=H.THETA_SAT,
        theta_gal_prof=None, theta_cib_prof=None,
        theta_sfr=np.array([1.0, FID["log10Mpeak"], FID["sigmaM0"],
                            FID["tau"], FID["zc"]]),
        theta_snu=np.array([10 ** FID["log10L0"], FID["beta_dust"],
                            FID["T0"], FID["alpha_dust"], 1.7]),
        theta_IR_hod=np.array([FID["log10Mmin_IR"], 0.4]),
        theta_sn_gI=np.zeros(3), theta_sn_II=np.zeros(6), hmalpha=1)

    for i, nu in enumerate(NU):
        d = meas[f"cl_A_{zk}_{nu}"][use]
        C = snr[f"cov_{zk}_{nu}"][np.ix_(use, use)]
        Ci = np.linalg.inv(C)
        # linear model: d = A*m + S   (A = CIB amplitude, S = flat shot noise)
        M = np.vstack([cgI[i][use], np.ones(use.sum())]).T
        MtCi = M.T @ Ci
        best = np.linalg.solve(MtCi @ M, MtCi @ d)
        r = d - M @ best
        chi2 = r @ Ci @ r
        dof = use.sum() - 2
        sn = np.sqrt(max(d @ Ci @ d, 0))
        results[(zk, nu)] = (best[0], chi2 / dof, sn)
        print(f"{zk:>9} {nu:>4} {best[0]:10.3e} {chi2/dof:9.2f} {dof:4d} {sn:9.1f}")

# --- exercise the Sampler on the real data vector for one bin ---------------
zk = ZBINS[1]
d = np.concatenate([meas[f"cl_A_{zk}_{nu}"][use] for nu in NU])
C = np.zeros((d.size, d.size))
n = use.sum()
for i, nu in enumerate(NU):          # block diagonal: per-frequency covariances
    C[i*n:(i+1)*n, i*n:(i+1)*n] = snr[f"cov_{zk}_{nu}"][np.ix_(use, use)]
n_pair = len(NU) * (len(NU) + 1) // 2
d_full = np.concatenate([d, np.zeros(n_pair * n)])
C_full = np.eye(d_full.size)
C_full[:d.size, :d.size] = C

s = Sampler(ana, d_full, C_full,
            theta_cen=H.THETA_CEN, theta_sat=H.THETA_SAT,
            n_nu=len(NU), ell_mask=use)
lo, hi = s.prior_bounds()
theta = np.array([FID[p] for p in SAMPLED_PARAMS] + [-2.0] * s.n_shot)
print(f"\n[sampler] ndim={s.ndim}  data={d_full.size}")
print(f"[sampler] log_prior  = {s.log_prior(theta):.4f}  (0 => inside the box)")
print(f"[sampler] loglike    = {s.loglike(theta):.4e}")
print(f"[sampler] logpost    = {s.logpost(theta):.4e}")
out = theta.copy(); out[0] = 99.0
print(f"[sampler] logpost outside prior = {s.logpost(out)}")
