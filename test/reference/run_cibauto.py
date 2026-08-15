"""
galCIB vs DopplerCIB: CIB x CIB.

This is the cleaner of the two comparisons:
  * no galaxy side at all -- no HOD, no nbar, no n(z)
  * the colour-correction conventions agree (DopplerCIB applies fc*cc to the
    CIB auto at CIB_halo.py:200, galCIB applies cc_II)
  * the reference output (2025-11-11) postdates the shared P(k)/HMF artifacts,
    unlike the gal x CIB reference (2024-08-01)

Reference: data/cl_cib_dopplercib_planck.npz, cls (6,6,99) over
[100,143,217,353,545,857] GHz, ells 100-2000.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from galCIB import (AnalysisModel, CIBModel, PkBuilder, SatProfile,  # noqa: E402
                    SFRModel, SnuModel, Survey, get_hod_model)
from galCIB.utils.io import load_my_filters  # noqa: E402

REPO = H.REPO
NU_ALL = [100, 143, 217, 353, 545, 857]

ells_ref, cls_ref = H.load_reference_cibauto()
print(f"[ref] cls {cls_ref.shape}  ells [{ells_ref[0]:.0f},{ells_ref[-1]:.0f}] "
      f"n={len(ells_ref)}")

z = H.load_reference_z()
zp, k, pk_ref = H.load_reference_pk()
mass = 10 ** np.arange(7, 16.005, 0.1)

filters = load_my_filters(str(REPO / "data" / "filters"), nu_obs=NU_ALL)
survey = Survey(z=z, pz=np.ones_like(z), mag_alpha=2.225, cib_filters=filters,
                ells=ells_ref, nside=2048)
cos = H.build_cosmology(z, k, pk_ref=pk_ref, mass=mass)
survey.compute_windows(cos)

prof = SatProfile(cos, profile_type="nfw")
hod_IR = get_hod_model("Zheng05", cos)
sfr = SFRModel(name="M21", hod=hod_IR, fsub=H.FSUB)
snu = SnuModel(name="M21", cosmo=cos, survey=survey,
               nu_prime=np.array(NU_ALL, dtype=float),
               m21_fdata=str(REPO / "data" / "filtered_snu_planck.fits"))
cib = CIBModel(hod_IR=hod_IR, sfr_model=sfr, snu_model=snu)
pk = PkBuilder(hod_model=get_hod_model("DESI-ELG", cos), cib_model=cib,
               gal_prof_model=prof, cib_prof_model=prof)
ana = AnalysisModel(survey=survey, pk3d=pk, limber_offset=0.0)

_, _, cII = ana.update_cl(
    theta_cen=H.THETA_CEN, theta_sat=H.THETA_SAT,
    theta_gal_prof=None, theta_cib_prof=None,
    theta_sfr=H.THETA_SFR_M21, theta_snu=None, theta_IR_hod=H.THETA_IR_HOD,
    theta_sn_gI=np.zeros(len(NU_ALL)),
    theta_sn_II=np.zeros(len(NU_ALL) * (len(NU_ALL) + 1) // 2),
    hmalpha=1)
print(f"[galCIB] cII {cII.shape}  (upper-triangle order)")

iu, ju = np.triu_indices(len(NU_ALL))
sel = (ells_ref > 200) & (ells_ref < 1500)
print(f"\n{'pair':>11} {'l=100':>9} {'l=600':>9} {'l=1200':>9} {'median':>9}")
ratios = {}
for idx, (i, j) in enumerate(zip(iu, ju)):
    r = cII[idx] / cls_ref[i, j]
    ratios[(NU_ALL[i], NU_ALL[j])] = np.median(r[sel])
    if NU_ALL[i] in (353, 545, 857) and NU_ALL[j] in (353, 545, 857):
        print(f"{NU_ALL[i]}x{NU_ALL[j]:<6} {r[0]:9.4f} "
              f"{r[np.argmin(abs(ells_ref-600))]:9.4f} "
              f"{r[np.argmin(abs(ells_ref-1200))]:9.4f} {np.median(r[sel]):9.4f}")

allr = np.array(list(ratios.values()))
print(f"\nall {len(allr)} frequency pairs, median ratio: "
      f"min={allr.min():.4f} median={np.median(allr):.4f} max={allr.max():.4f}")
print("does the reference include shot noise? (DopplerCIB adds a flat, "
      "diagonal-only shot term)")
for nu in (353, 545, 857):
    i = NU_ALL.index(nu)
    d = cls_ref[i, i]
    print(f"  {nu}x{nu}: ref C_l at l=100 -> {d[0]:.4e}, "
          f"at l=2000 -> {d[-1]:.4e}, ratio {d[0]/d[-1]:.2f}")

# ---- shot noise: DopplerCIB adds a flat, DIAGONAL-ONLY term ----------------
# CIB_halo.py:250-281. Values for 217/353/545/857 come from the bestfit file
# rows 4:8; 100 and 143 are hardcoded as 1.3x Matt's model. Not colour-corrected
# (cl_cibtot = oneh + twoh + shot, and fcxcc is applied only to oneh/twoh).
bf = np.loadtxt(H.DOPPLER / "data_files" /
                "one_halo_bestfit_allcomponents_lognormal_sigevol_1p5zcutoff"
                "_nolens_onlyautoshotpar_no3000_gaussian600n857n1200_planck"
                "_spire_hmflog10.txt")
sa = np.array([1.3*0.116689509208305475, 1.3*0.8714424869942087,
               14., 357., 2349., 7407.])
sa[2:] = bf[4:8, 0]
print(f"\nDopplerCIB shot values (Jy^2/sr): "
      + ", ".join(f"{n}:{s:.1f}" for n, s in zip(NU_ALL, sa)))

print(f"\n{'pair':>11} {'no shot':>9} {'+shot':>9}")
for idx, (i, j) in enumerate(zip(iu, ju)):
    shot = sa[i] if i == j else 0.0          # diagonal only
    r_no = np.median((cII[idx] / cls_ref[i, j])[sel])
    r_sh = np.median(((cII[idx] + shot) / cls_ref[i, j])[sel])
    if NU_ALL[i] in (353, 545, 857) and NU_ALL[j] in (353, 545, 857):
        print(f"{NU_ALL[i]}x{NU_ALL[j]:<6} {r_no:9.4f} {r_sh:9.4f}")

allr = []
for idx, (i, j) in enumerate(zip(iu, ju)):
    shot = sa[i] if i == j else 0.0
    allr.append(np.median(((cII[idx] + shot) / cls_ref[i, j])[sel]))
allr = np.array(allr)
print(f"\nALL 21 pairs with shot noise: min={allr.min():.4f} "
      f"median={np.median(allr):.4f} max={allr.max():.4f}")
