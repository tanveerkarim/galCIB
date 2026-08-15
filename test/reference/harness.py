"""
Shared setup for validating galCIB against DopplerCIB.

Ground truth is DopplerCIB. Rather than re-running it, we compare against the
reference outputs already saved in `data/`:

    cl_cib_dopplercib_planck.npz                    CIB x CIB, (6,6,99)
    PlanckCIB{353,545,857}GHzxDESI_ELGgalaxy_rl*.txt  gal x CIB, cols ell,1h,2h,shot,total

The gal x CIB files store the 1-halo and 2-halo terms *separately*, so a
mismatch localises to one term rather than only showing up in the total.

Both codes must be driven from identical inputs or the comparison is
meaningless. In particular galCIB's internally computed P(k) differs from the
pickle DopplerCIB was fed by up to 3.6%, k-dependently (different camb version),
so `build_cosmology` injects the reference P(k). The HMF needs no such treatment
-- galCIB reproduces the saved DopplerCIB HMF exactly.

Compatibility settings that disable galCIB-only features for the diff
(see the plan): limber_offset=0, hmalpha=1, no W_mu, pure-NFW profile shared
between galaxies and CIB, M21 tabulated SED, Ncen_IR = 1.
"""

import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DOPPLER = REPO.parent / "DopplerCIB"

# --- DopplerCIB configuration behind the saved outputs ----------------------
# driver_cibxgal.py:10,68,57 ; CIB_halo.py:41-47 ; Gal_halo.py:29-69
MASS = 10 ** np.arange(7, 16.005, 0.1)
ELLS = np.linspace(100, 5000, 99)
NU_OBS = [353, 545, 857]
FSUB = 0.134

# CIB_halo.py:41-47 -- "based on Y23 best fit of M21"
#   etamax, Meffmax, tau, sigmaMh, z_c
THETA_SFR_M21 = np.array([0.42, 12.94, 1.75, 1.17, 1.5])

# Gal_halo.py:29-69, DESI_ELG branch
#   Ncen_mHMQ: gamma, log10Mc, sigmaM, Ac
THETA_CEN = np.array([3.28, 11.49, 0.45, 0.1])
#   Nsat_ELG: As, M0, M1, alpha_sat
THETA_SAT = np.array([0.38, 10**11.14, 10**13.0, 0.59])

# DopplerCIB has no IR HOD -- djc_dlnMh carries no Ncen factor (CIB_halo.py:98).
# Reproduce that by driving Zheng05 Ncen to 1 with a tiny Mmin.
THETA_IR_HOD = np.array([1.0, 0.4])


def load_reference_z():
    """The 210-point redshift grid both codes must share."""
    return np.loadtxt(DOPPLER / "data_files" / "redshifts.txt")


def load_reference_pk():
    """Return (z, k, pk) with pk shaped (Nk, Nz), as galCIB stores it."""
    d = pickle.load(open(REPO / "data" / "plin_unit_Mpc_DopplerCIB.p", "rb"))
    return np.asarray(d["z"]), np.asarray(d["k"]), np.asarray(d["pk"]).T


def load_reference_dndz():
    """DESI ELG dn/dz actually used for the saved reference outputs.

    Gal_halo.py:113 has a live "#FIXME: temporary test" that overrides
    data_files/dndz_DESI_ELG.txt with a pickle, and the commented-out lines
    immediately below it read `zrange` and `dndz.mean(axis=0)` -- which is
    exactly the structure of data/gal/dndz_extended.p (1000 realisations x 30
    z-bins). Use the mean over realisations.
    """
    d = pickle.load(open(REPO / "data" / "gal" / "dndz_extended.p", "rb"))
    return np.asarray(d["zrange"]), np.asarray(d["dndz"]).mean(axis=0)


def load_shared_grid():
    """The grid the shared DopplerCIB<->galCIB inputs pin down.

    Not a free choice: tmp/hmf.npy and tmp/bnu.npy are (100, 20) and
    tmp/pz.npz['z'] is np.linspace(0.5, 1.5, 20), and galCIB reproduces
    tmp/hmf.npy bit-exactly with np.logspace(7, 15, 100) on that redshift grid.

    Returns (z, pz, mass); pz is raw -- normalise before handing it to
    compute_Wg, which does not (see load_reference_dndz).
    """
    d = np.load(REPO / "tmp" / "pz.npz")
    return np.asarray(d["z"]), np.asarray(d["pz"]), np.logspace(7, 15, 100)


def load_reference_galxcib():
    """gal x CIB reference regenerated from the CURRENT DopplerCIB.

    Produced by test/reference/gen_galxcib_reference.py. Keys are
    '{variant}_{term}_{nu}' with variant in {shared, internal},
    term in {1h, 2h, shot}, nu in {353, 545, 857}, plus ells/z/mass.
    """
    return np.load(REPO / "data" / "cl_galxcib_dopplercib_planck.npz")


def load_reference_cross(nu, rl="1.0"):
    """DEPRECATED gal x CIB reference: dict of ell, 1h, 2h, shot, total.

    These files are from 2024-08-01 and predate every shared input by 9-15
    months; they were generated on a different z grid, mass grid and n(z), so
    diffing against them says nothing about galCIB (THEORY_VALIDATION.md 3b).
    Kept only so run_tier1.py can reproduce the retraction.
    Use load_reference_galxcib() instead.
    """
    t = np.loadtxt(REPO / "data" / f"PlanckCIB{nu}GHzxDESI_ELGgalaxy_rl{rl}.txt")
    return dict(ell=t[:, 0], oneh=t[:, 1], twoh=t[:, 2], shot=t[:, 3], total=t[:, 4])


def load_reference_cibauto():
    """CIB x CIB reference, (6,6,99) over [100,143,217,353,545,857]."""
    z = np.load(REPO / "data" / "cl_cib_dopplercib_planck.npz")
    return np.asarray(z["ells"]), np.asarray(z["cls"])


def build_cosmology(z, k, pk_ref=None, mass=MASS):
    """galCIB Cosmology with the reference P(k) injected.

    Injection matters: sigma(R) -> nu -> bias and c(M,z) all read
    `cosmo.pk_grid`, so overriding it propagates consistently through the halo
    model. The HMF comes from colossus independently and already matches.
    """
    from galCIB import Cosmology

    cos = Cosmology(z, k, mass, colossus_cosmo_name="planck18", use_little_h=False)
    if pk_ref is not None:
        assert pk_ref.shape == cos.pk_grid.shape, (pk_ref.shape, cos.pk_grid.shape)
        cos.pk_grid = pk_ref
    return cos
