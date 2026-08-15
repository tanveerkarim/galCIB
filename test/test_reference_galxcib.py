"""
Regression guard on the gal x CIB agreement with DopplerCIB.

THEORY_VALIDATION.md 3b closed this comparison at 0.27%, with the residual
fully accounted for by colossus (Om0=0.31110) vs astropy (Om0=0.30966) entering
the baryon accretion rate. The tolerances here are set just outside that: 1% on
the normalisation, 0.1% on the flatness in ell and in frequency.

Flatness is the stricter and more informative assertion. A normalisation shift
can come from a convention (and one does); an ell- or nu-dependent drift cannot,
so it would mean the halo model itself had changed.

Skipped when data/cl_galxcib_dopplercib_planck.npz is absent -- regenerate with
    python test/reference/gen_galxcib_reference.py
which needs ../DopplerCIB checked out alongside this repo.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REF_DIR = Path(__file__).resolve().parent / "reference"
sys.path.insert(0, str(REF_DIR))

import run_galxcib as R  # noqa: E402

pytestmark = pytest.mark.skipif(
    not R.REFERENCE.exists(),
    reason=f"{R.REFERENCE.name} not generated; run gen_galxcib_reference.py",
)


@pytest.fixture(scope="module")
def ratios():
    return R.compare(verbose=False)


@pytest.mark.parametrize("nu", R.NU_OBS)
@pytest.mark.parametrize("term", ["1h", "2h", "tot"])
def test_normalisation_within_1pct(ratios, nu, term):
    r = ratios[("shared", nu)]
    med = np.median(r[term][r["sel"]])
    assert abs(med - 1.0) < 0.01, f"{term} at {nu} GHz: median ratio {med:.4f}"


@pytest.mark.parametrize("nu", R.NU_OBS)
@pytest.mark.parametrize("term", ["1h", "2h"])
def test_flat_in_ell(ratios, nu, term):
    """The ratio must not drift with scale -- that is the shape check."""
    r = ratios[("shared", nu)][term]
    spread = r.max() / r.min() - 1.0
    assert spread < 1e-3, f"{term} at {nu} GHz drifts {spread:.2%} across ell"


@pytest.mark.parametrize("term", ["1h", "2h", "tot"])
def test_flat_in_frequency(ratios, term):
    """A frequency-dependent residual would point at the SED or colour terms."""
    med = np.array([np.median(ratios[("shared", nu)][term]
                              [ratios[("shared", nu)]["sel"]])
                    for nu in R.NU_OBS])
    assert med.max() / med.min() - 1.0 < 1e-3, f"{term} spread across nu: {med}"
