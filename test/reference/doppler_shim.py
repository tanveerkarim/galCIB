"""
Run DopplerCIB in-process without editing it.

DopplerCIB is the ground truth for this validation, so it stays pristine: every
path it hardcodes is redirected here instead. The stale prefixes below are the
literal strings in the DopplerCIB source (two generations of them -- an older
`/Users/tkarim/...` layout and a newer `Macintosh HD` copy of the same tree).

Patches numpy.load / numpy.loadtxt / glob.glob / astropy.io.fits.open and
builtins.open, because DopplerCIB reads files through all five. Relative paths
like 'data_files/...' are handled by chdir'ing into the DopplerCIB directory.
"""

import builtins
import glob as _glob
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
DOPPLER = REPO.parent / "DopplerCIB"

# Longest first: the "Macintosh HD" prefixes contain the shorter ones.
_MAP = [
    ("/Users/tanveerk/research/Macintosh HD/Users/tkarim/research/galCIB", str(REPO)),
    ("/Users/tanveerk/research/Macintosh HD/Users/tkarim/research/DopplerCIB", str(DOPPLER)),
    ("/Users/tkarim/Documents/research/DopplerCIB", str(DOPPLER)),
    ("/Users/tkarim/research/galCIB", str(REPO)),
    ("/Users/tkarim/research/DopplerCIB", str(DOPPLER)),
]

_rewrites = {}


def _fix(p):
    if not isinstance(p, (str, os.PathLike)):
        return p
    s = str(p)
    for old, new in _MAP:
        if s.startswith(old):
            out = new + s[len(old):]
            _rewrites[s] = out
            return out
    return p


def _install_numpy_scipy_aliases():
    """Restore names DopplerCIB uses that newer numpy/scipy dropped.

    Pure aliasing -- no behaviour is changed. `even='avg'` was scipy's default
    for an even number of samples, so dropping the kwarg on a modern simpson
    would silently change the answer; simpson's current default is 'avg'
    equivalent, so the kwarg is simply removed.
    """
    import scipy.integrate as intg

    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    if not hasattr(intg, "simps"):
        _simpson = intg.simpson

        def simps(y, *a, **k):
            k.pop("even", None)
            return _simpson(y, *a, **k)

        intg.simps = simps


def install():
    """Patch the file-reading entry points and chdir into DopplerCIB."""
    _install_numpy_scipy_aliases()
    _np_load, _np_loadtxt = np.load, np.loadtxt
    _glob_glob, _fits_open, _open = _glob.glob, fits.open, builtins.open

    np.load = lambda f, *a, **k: _np_load(_fix(f), *a, **k)
    np.loadtxt = lambda f, *a, **k: _np_loadtxt(_fix(f), *a, **k)
    _glob.glob = lambda p, *a, **k: _glob_glob(_fix(p), *a, **k)
    fits.open = lambda f, *a, **k: _fits_open(_fix(f), *a, **k)
    builtins.open = lambda f, *a, **k: _open(_fix(f), *a, **k)

    os.chdir(DOPPLER)
    import sys
    if str(DOPPLER) not in sys.path:
        sys.path.insert(0, str(DOPPLER))


def rewrites():
    """Paths actually redirected, for the provenance record."""
    return dict(_rewrites)
