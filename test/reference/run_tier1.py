"""
Tier 1: reproduce the saved DopplerCIB gal x CIB spectra with galCIB.

Compares the 1-halo and 2-halo terms separately, since the reference files
store them separately -- a mismatch then localises to one term.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from galCIB import (  # noqa: E402
    AnalysisModel, CIBModel, Cosmology, PkBuilder, SatProfile,
    SFRModel, SnuModel, Survey, get_hod_model,
)
from galCIB.utils.io import load_my_filters  # noqa: E402

REPO = H.REPO

z = H.load_reference_z()
zp, k, pk_ref = H.load_reference_pk()

# dn/dz is tabulated on a coarse 20-point grid over 0 < z < 1.9; interpolate
# onto the shared 210-point grid and zero outside its support.
dz_z, dz_p = H.load_reference_dndz()
pz = np.interp(z, dz_z, dz_p, left=0.0, right=0.0)
# galCIB's compute_Wg does NOT normalise (survey/window.py: "wgal = pz/dchi_dz"),
# whereas DopplerCIB divides by N = simpson(dn/dz, z) (Gal_halo.py:211).
# Supply a normalised p(z) so the two windows agree.
from scipy.integrate import simpson
pz = pz / simpson(pz, x=z)

filters = load_my_filters(str(REPO / "data" / "filters"), nu_obs=H.NU_OBS)
survey = Survey(z=z, pz=pz, mag_alpha=2.225, cib_filters=filters,
                ells=H.ELLS, nside=1024)

cos = H.build_cosmology(z, k, pk_ref=pk_ref)
survey.compute_windows(cos)
print(f"[cfg] z {z.shape}  k {k.shape}  M {cos.Mh.shape}  ells {H.ELLS.shape}")

elg_hod = get_hod_model("DESI-ELG", cos)
hod_IR = get_hod_model("Zheng05", cos)
# DopplerCIB uses a pure NFW profile for both satellites and subhalos --
# there is no exponential/mixed profile anywhere in that code.
prof = SatProfile(cos, profile_type="nfw")

sfr = SFRModel(name="M21", hod=hod_IR, fsub=H.FSUB)
snu = SnuModel(name="M21", cosmo=cos, survey=survey,
               nu_prime=np.array(H.NU_OBS, dtype=float),
               m21_fdata=str(REPO / "data" / "filtered_snu_planck.fits"))
cib = CIBModel(hod_IR=hod_IR, sfr_model=sfr, snu_model=snu)

pk = PkBuilder(hod_model=elg_hod, cib_model=cib,
               gal_prof_model=prof, cib_prof_model=prof)

# limber_offset=0 reproduces DopplerCIB's k = ell/chi
ana = AnalysisModel(survey=survey, pk3d=pk, limber_offset=0.0)

cgg, cgI, cII = ana.update_cl(
    theta_cen=H.THETA_CEN, theta_sat=H.THETA_SAT,
    theta_gal_prof=None, theta_cib_prof=None,
    theta_sfr=H.THETA_SFR_M21, theta_snu=None, theta_IR_hod=H.THETA_IR_HOD,
    theta_sn_gI=np.zeros(len(H.NU_OBS)),
    theta_sn_II=np.zeros(len(H.NU_OBS) * (len(H.NU_OBS) + 1) // 2),
    hmalpha=1,
)
print(f"[out] cgg {cgg.shape}  cgI {cgI.shape}  cII {cII.shape}")

print(f"\n{'nu':>5} {'ell':>7} {'galCIB':>12} {'DopplerCIB':>12} {'ratio':>8}")
for i, nu in enumerate(H.NU_OBS):
    ref = H.load_reference_cross(nu)
    # compare against 1h+2h: galCIB was run with theta_sn_gI = 0, and the
    # reference `total` column additionally contains a shot-noise term.
    sig = ref["oneh"] + ref["twoh"]
    for j in (0, 10, 30, 60):
        print(f"{nu:>5} {ref['ell'][j]:7.0f} {cgI[i, j]:12.4e} "
              f"{sig[j]:12.4e} {cgI[i, j] / sig[j]:8.4f}")
    m = (ref["ell"] > 200) & (ref["ell"] < 2000)
    print(f"      median ratio (200<l<2000): "
          f"{np.median(cgI[i][m] / sig[m]):.4f}")

# ---- split 1-halo vs 2-halo -------------------------------------------------
# update_cl only returns totals, so project the separate P(k) terms by hand.
print(f"\n--- 1-halo / 2-halo split (galCIB / DopplerCIB) ---")
print(f"{'nu':>5} {'ell':>7} {'1h ratio':>10} {'2h ratio':>10} {'cc_gI':>7}")
for i, nu in enumerate(H.NU_OBS):
    ref = H.load_reference_cross(nu)
    p1 = ana._kPk_interpolator(pk.pk_gI_1h[i])
    p2 = ana._kPk_interpolator(pk.pk_gI_2h[i])
    c1 = ana.compute_cl(p1, ana.Wg, ana.Wcib)
    c2 = ana.compute_cl(p2, ana.Wg, ana.Wcib)
    for j in (0, 10, 30, 60):
        print(f"{nu:>5} {ref['ell'][j]:7.0f} {c1[j]/ref['oneh'][j]:10.4f} "
              f"{c2[j]/ref['twoh'][j]:10.4f} {float(ana.cc_gI[i,0]):7.4f}")
    m = (ref["ell"] > 200) & (ref["ell"] < 2000)
    print(f"      median: 1h={np.median(c1[m]/ref['oneh'][m]):.4f}  "
          f"2h={np.median(c2[m]/ref['twoh'][m]):.4f}")

# ---- test the u_cen hypothesis ---------------------------------------------
# DopplerCIB (CIBxGal_halo.py:93-94) gives centrals an NFW profile in the cross:
#   galterm = Ncen*u + Nsat*u
# galCIB uses u_cen = 1:
#   galterm = Ncen + Nsat*u
# At high k, u -> 0, so DopplerCIB's galterm vanishes while galCIB's tends to
# Ncen. That predicts a 2-halo ratio that GROWS with ell -- which is what we see.
from scipy.integrate import simpson as _simp
print("\n--- u_cen hypothesis: recompute Ig with DopplerCIB's convention ---")
gal_u = pk.gal_prof_model.u
Ig_dopp = _simp(pk.hmfxbias * (pk.ncen * gal_u + pk.nsat_u), dx=pk.dlog10Mh, axis=1)
print(f"{'nu':>5} {'ell':>7} {'2h ratio (u_cen=1)':>19} {'2h ratio (u_cen=u)':>19}")
for i, nu in enumerate(H.NU_OBS):
    ref = H.load_reference_cross(nu)
    p2_orig = pk.Ig * (pk.Icib[i] * pk.cosmo.pk_grid) / pk.nbar
    p2_dopp = Ig_dopp * (pk.Icib[i] * pk.cosmo.pk_grid) / pk.nbar
    c_orig = ana.compute_cl(ana._kPk_interpolator(p2_orig), ana.Wg, ana.Wcib)
    c_dopp = ana.compute_cl(ana._kPk_interpolator(p2_dopp), ana.Wg, ana.Wcib)
    for j in (0, 30, 60):
        print(f"{nu:>5} {ref['ell'][j]:7.0f} {c_orig[j]/ref['twoh'][j]:19.4f} "
              f"{c_dopp[j]/ref['twoh'][j]:19.4f}")
