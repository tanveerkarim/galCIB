"""
Regenerate the gal x CIB reference by running DopplerCIB with the shared inputs.

Why this exists
---------------
The checked-in reference (data/PlanckCIB*GHzxDESI_ELGgalaxy_rl*.txt, 2024-08-01)
predates the harmonised inputs by 9-15 months and was produced on a different
grid, so a disagreement against it says nothing about galCIB. This script
re-runs the *current* DopplerCIB on the grid the shared files define.

The shared grid is not a free choice -- it is recorded in the files themselves:

    tmp/hmf.npy, tmp/bnu.npy   (100, 20)   -> 100 masses, 20 redshifts
    tmp/pz.npz  ['z']          (20,)       -> np.linspace(0.5, 1.5, 20)

and galCIB reproduces tmp/hmf.npy bit-exactly with
Mh = np.logspace(7, 15, 100) on that redshift grid, which pins the mass grid.

DopplerCIB itself is never edited: doppler_shim redirects its hardcoded paths.

Two variants are written, because DopplerCIB is internally inconsistent about
which halo mass function the cross uses. Gal_halo.py:18-19 injects the shared
arrays, but CIBxgal.__init__ calls Cib_halo.__init__ *after*
ProfHODMore15.__init__ and Cib_halo.py:51-55 overwrites both with its own. So:

    internal : what DopplerCIB computes if you just run it (colossus HMF on
               its own grid, bias interpolated from the 210-point z grid)
    shared   : hmfmz/biasmz forced to tmp/hmf.npy and tmp/bnu.npy

Output: data/cl_galxcib_dopplercib_planck.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doppler_shim  # noqa: E402

REPO = doppler_shim.REPO
doppler_shim.install()

from CIBxGal_halo import CIBxgal  # noqa: E402
from input_var_cibmean import data_var_iv  # noqa: E402
from cosmo_related import cosmo_var_iv  # noqa: E402

# --- the shared grid --------------------------------------------------------
MASS = np.logspace(7, 15, 100)
_pz = np.load(REPO / "tmp" / "pz.npz")
Z = _pz["z"]
ELLS = np.linspace(100, 5000, 99)
NU_ALL = [100.0, 143.0, 217.0, 353.0, 545.0, 857.0]
NU_SAVE = [353, 545, 857]          # indices 3, 4, 5
R_L = 1.0

# driver_cibxgal.py:56-76
exp = {
    "name": "Planck",
    "do_cibmean": 0,
    "cc": np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995]),
    "fc": np.ones(6),
    "snuaddr": "data_files/filtered_snu_planck.fits",
    "nu0": np.array(NU_ALL),
    "ell": ELLS,
    "fwhm": np.array([9.651, 7.248, 4.990, 4.818, 4.682, 4.325]),
    "sensitivity": np.sqrt(np.array([58.0, 26.929, 72.0, 305.0, 369.0, 369.0])),
    "cibpar_resfile": "data_files/one_halo_bestfit_allcomponents_lognormal"
                      "_sigevol_1p5zcutoff_nolens_onlyautoshotpar_no3000"
                      "_gaussian600n857n1200_planck_spire_hmflog10.txt",
    "nu_string": ["100", "143", "217", "353", "545", "857"],
}

print(f"[grid] M {MASS.shape}  z {Z.shape} [{Z[0]:.2f},{Z[-1]:.2f}]  "
      f"ells {ELLS.shape}")

driver = data_var_iv(exp)
uni = cosmo_var_iv(MASS, Z, do_powerspec=1)
cross = CIBxgal(driver, uni, "DESI_ELG", R_L)

# Sanity: does DopplerCIB's own HMF match the shared one it was supposed to use?
hmf_shared = np.load(REPO / "tmp" / "hmf.npy")
bnu_shared = np.load(REPO / "tmp" / "bnu.npy")
r_h = cross.hmfmz / hmf_shared
r_b = cross.biasmz / bnu_shared
print(f"[check] DopplerCIB hmf  / shared: {r_h.min():.6f} - {r_h.max():.6f}")
print(f"[check] DopplerCIB bias / shared: {r_b.min():.6f} - {r_b.max():.6f}")

out = {"ells": ELLS, "z": Z, "mass": MASS, "nu": np.array(NU_SAVE),
       "r_l": R_L, "hmf_ratio_span": np.array([r_h.min(), r_h.max()]),
       "bias_ratio_span": np.array([r_b.min(), r_b.max()])}

for variant in ("internal", "shared"):
    if variant == "shared":
        cross.hmfmz = hmf_shared
        cross.biasmz = bnu_shared
    oneh = cross.cibgalcross_cell_1h()
    twoh = cross.cibgalcross_cell_2h()
    shot = cross.cibgalcross_cell_shot()
    for i, nu in enumerate(NU_SAVE):
        j = NU_ALL.index(float(nu))
        out[f"{variant}_1h_{nu}"] = oneh[j]
        out[f"{variant}_2h_{nu}"] = twoh[j]
        out[f"{variant}_shot_{nu}"] = shot[j]
    print(f"[{variant}] 353GHz 1h(l=100)={oneh[3, 0]:.4e}  "
          f"2h(l=100)={twoh[3, 0]:.4e}  shot={shot[3, 0]:.4e}")

dest = REPO / "data" / "cl_galxcib_dopplercib_planck.npz"
np.savez(dest, **out)
print(f"\n[saved] {dest}")
print("[paths redirected]")
for old, new in doppler_shim.rewrites().items():
    print(f"  {old}\n    -> {new}")
