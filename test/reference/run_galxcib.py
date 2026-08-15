"""
galCIB vs DopplerCIB: gal x CIB, against the FRESH reference.

Run test/reference/gen_galxcib_reference.py first -- it produces
data/cl_galxcib_dopplercib_planck.npz by re-running the current DopplerCIB on
the shared grid. The old checked-in PlanckCIB*.txt files (2024-08-01) predate
the shared inputs and are not comparable; see THEORY_VALIDATION.md 3b.

Both sides are driven from the same shared inputs:
    z    = tmp/pz.npz['z']              np.linspace(0.5, 1.5, 20)
    p(z) = tmp/pz.npz['pz']             normalised, since compute_Wg does not
    M    = np.logspace(7, 15, 100)      reproduces tmp/hmf.npy bit-exactly
    P(k) = data/plin_unit_Mpc_DopplerCIB.p, interpolated in z as DopplerCIB does

Compared term by term (1-halo and 2-halo separately) with galCIB's extras
switched off, so any residual is a genuine finding:
    limber_offset = 0    -> k = ell/chi, DopplerCIB's convention
    hmalpha = 1          -> plain 1h + 2h sum
    no magnification     -> compute_cl called directly, bypassing CgI_tot
    no colour correction -> DopplerCIB does not apply fc*cc in the cross
                            (CIBxGal_halo.py:90-115), unlike its CIB auto
    pure NFW profile shared by galaxies and CIB
    Ncen_IR = 1          -> DopplerCIB has no IR HOD

Importable: `compare()` returns the ratios, so test/test_reference_galxcib.py
can assert on them without duplicating the setup.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from galCIB import (AnalysisModel, CIBModel, PkBuilder, SatProfile,  # noqa: E402
                    SFRModel, SnuModel, Survey, get_hod_model)
from galCIB.utils.io import load_my_filters  # noqa: E402

REPO = H.REPO
NU_OBS = [353, 545, 857]
REFERENCE = REPO / "data" / "cl_galxcib_dopplercib_planck.npz"


def compare(verbose=True):
    """Return {(variant, nu): {'1h','2h','tot'}} of galCIB/DopplerCIB ratios."""
    ref = np.load(REFERENCE)
    ells, z, mass = ref["ells"], ref["z"], ref["mass"]
    if verbose:
        print(f"[ref] z {z.shape} [{z[0]:.2f},{z[-1]:.2f}]  M {mass.shape}  "
              f"ells {ells.shape}  hmf ratio {ref['hmf_ratio_span']}")

    # p(z): the shared file. galCIB's compute_Wg does not normalise, DopplerCIB
    # divides by N = simpson(dn/dz, z) (Gal_halo.py:211,219), so normalise here.
    pz = np.load(REPO / "tmp" / "pz.npz")["pz"]
    pz = pz / simpson(pz, x=z)

    # P(k): the same pickle DopplerCIB reads, interpolated onto the 20-point
    # grid exactly as cosmo_related.py:55 does (linear in z, per k).
    zp, k, pk_full = H.load_reference_pk()          # (Nk, Nz=210)
    pk_ref = np.array([np.interp(z, zp, pk_full[i]) for i in range(len(k))])

    filters = load_my_filters(str(REPO / "data" / "filters"), nu_obs=NU_OBS)
    survey = Survey(z=z, pz=pz, mag_alpha=2.225, cib_filters=filters,
                    ells=ells, nside=1024)
    cos = H.build_cosmology(z, k, pk_ref=pk_ref, mass=mass)
    survey.compute_windows(cos)

    # The HMF is the single most important shared input: confirm it first.
    r_h = cos.hmf_grid / np.load(REPO / "tmp" / "hmf.npy")
    if verbose:
        print(f"[check] galCIB hmf / shared: {r_h.min():.8f} - {r_h.max():.8f}")

    prof = SatProfile(cos, profile_type="nfw")
    hod_IR = get_hod_model("Zheng05", cos)
    sfr = SFRModel(name="M21", hod=hod_IR, fsub=H.FSUB)
    snu = SnuModel(name="M21", cosmo=cos, survey=survey,
                   nu_prime=np.array(NU_OBS, dtype=float),
                   m21_fdata=str(REPO / "data" / "filtered_snu_planck.fits"))
    cib = CIBModel(hod_IR=hod_IR, sfr_model=sfr, snu_model=snu)
    pk = PkBuilder(hod_model=get_hod_model("DESI-ELG", cos), cib_model=cib,
                   gal_prof_model=prof, cib_prof_model=prof)
    ana = AnalysisModel(survey=survey, pk3d=pk, limber_offset=0.0)

    ana.update_cl(theta_cen=H.THETA_CEN, theta_sat=H.THETA_SAT,
                  theta_gal_prof=None, theta_cib_prof=None,
                  theta_sfr=H.THETA_SFR_M21, theta_snu=None,
                  theta_IR_hod=H.THETA_IR_HOD,
                  theta_sn_gI=np.zeros(len(NU_OBS)),
                  theta_sn_II=np.zeros(len(NU_OBS) * (len(NU_OBS) + 1) // 2),
                  hmalpha=1)

    # 1h and 2h projected by hand: update_cl returns only the total, and it
    # carries magnification and the colour correction that DopplerCIB lacks.
    cl = {}
    for i, nu in enumerate(NU_OBS):
        cl[nu] = (ana.compute_cl(ana._kPk_interpolator(pk.pk_gI_1h[i]),
                                 ana.Wg, ana.Wcib),
                  ana.compute_cl(ana._kPk_interpolator(pk.pk_gI_2h[i]),
                                 ana.Wg, ana.Wcib))

    sel = (ells > 200) & (ells < 2000)
    j6 = int(np.argmin(abs(ells - 600)))
    out = {}
    for variant in ("shared", "internal"):
        if verbose:
            label = ("shared hmf+bias forced" if variant == "shared"
                     else "DopplerCIB internal")
            print(f"\n=== variant: {variant} ({label}) ===")
            print(f"{'nu':>5} {'1h l=100':>9} {'1h l=600':>9} {'1h med':>9} "
                  f"{'2h l=100':>9} {'2h l=600':>9} {'2h med':>9} {'tot med':>9}")
        for nu in NU_OBS:
            c1, c2 = cl[nu]
            r1h = ref[f"{variant}_1h_{nu}"]
            r2h = ref[f"{variant}_2h_{nu}"]
            a1, a2 = c1 / r1h, c2 / r2h
            at = (c1 + c2) / (r1h + r2h)
            out[(variant, nu)] = {"1h": a1, "2h": a2, "tot": at, "sel": sel}
            if verbose:
                print(f"{nu:>5} {a1[0]:9.4f} {a1[j6]:9.4f} "
                      f"{np.median(a1[sel]):9.4f} {a2[0]:9.4f} {a2[j6]:9.4f} "
                      f"{np.median(a2[sel]):9.4f} {np.median(at[sel]):9.4f}")
    return out


if __name__ == "__main__":
    compare()
