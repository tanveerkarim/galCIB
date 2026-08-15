"""
Measure DESI DR2 ELG x Lenz19 CIB angular cross-power spectra with NaMaster.

Galaxies are split into tomographic bins of width dz = 0.2 and cross-correlated
against the Lenz+19 CIB maps at 353, 545 and 857 GHz. The galaxy field is built
three ways so the catalog-based scheme can be validated against a map-based one:

  A  catalog  -- NmtFieldCatalogClustering, footprint from the random catalog.
  B  hybrid   -- NmtFieldCatalogClustering, footprint from a pixelised random
                 density map (`mask=`). No mask shot noise.
  C  map      -- NmtField on a pixelised overdensity map, randoms used as the
                 weight map (NOT a binary footprint).

Only *decoupled* C_ells from these routes are comparable. The coupled pseudo-C_ell
of a catalog field carries the unnormalised mask scale sum(w_rand) ~ 1.3e8, so it
sits ~1e16 above a map-based pseudo-C_ell; NmtWorkspace absorbs this exactly.

The primary validation is the gal x CIB *cross*-spectrum, which carries no shot
noise at all, rather than the auto-spectrum.

Usage
-----
    python analysis/measure_gal_cib.py --stage cats     # rotate + cache catalogs
    python analysis/measure_gal_cib.py --smoke          # one (zbin, nu), timed
    python analysis/measure_gal_cib.py --routes A C     # full run
"""

import argparse
import os
from pathlib import Path
from time import time

import healpy as hp
import numpy as np
import pandas as pd
import pymaster as nmt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent

CIB_FOLDER = REPO / "data" / "cib" / "cib-lenz19-data"
DR2_FOLDER = REPO / "data" / "dr2"
OUT_FOLDER = REPO / "data" / "measurements" / "gal_cib"
CACHE = OUT_FOLDER / "cache"

NULIST = ["353", "545", "857"]
NHI = "4.0e+20_gp40"

NSIDE = 2048
LMAX = 2000       # field band limit; Lenz 5' Bl_eff ~ 0.83 at l = 1000
LMAX_MASK = 4000  # 2 x LMAX -- convergence-tested, see module docstring
NLB = 50          # -> 40 bandpowers

# Usable range for plotting/fitting. The measurement is deliberately computed
# out to LMAX and cut here rather than being computed at a lower lmax: it costs
# ~40s either way, and keeping headroom above the cut avoids mode-coupling
# truncation bias at the top of the range you actually use. Above ~600 the
# tomographic bins are shot-noise dominated (0.115 galaxies per nside-2048
# pixel) with little 1-halo power left, and the bandpowers scatter about zero.
ELL_USE_MAX = 600

# Resolution at which the random density (completeness) map is estimated before
# being upgraded to NSIDE. The randoms are far too sparse to be binned directly
# at 2048: in bin 0.8-1.0 that gives 1.51 randoms per occupied pixel, 62% of
# them singletons, and it recovers only fsky = 0.163 of the true fsky ~ 0.31
# footprint. At 512 the same randoms give 12.8 per pixel and fsky = 0.3075.
# Measured convergence of route B onto route A over 100 < l < 500, per-bin
# randoms: nside 256 -> 6.1%, 512 -> 1.15%, 1024 -> 0.75%. 512 is the compromise:
# route C divides pixel-by-pixel by this mask, so it wants the better-sampled
# 12.8 counts/pixel (28% Poisson scatter) over 1024's 3.5 (53%).
MASK_NSIDE = 512

DZ = 0.2          # tomographic bin width; 0.4 and 0.8 also supported
Z_MIN, Z_MAX = 0.8, 1.6
NRAN_FILES = 2    # number of random catalogs to use

NPIX = hp.nside2npix(NSIDE)
CHUNK = 5_000_000  # rows per SkyCoord block

Z_EDGES = np.round(np.arange(Z_MIN, Z_MAX + DZ / 2, DZ), 1)


def zkey(i):
    """Bin label for tomographic bin `i`. Single source of truth: data and
    randoms must never build these independently (the notebook rounded only
    the data edges and produced '1.4-1.5999999999999999' for the randoms)."""
    return f"{Z_EDGES[i]}-{Z_EDGES[i + 1]}"


ZKEYS = [zkey(i) for i in range(len(Z_EDGES) - 1)]


def configure(dz=None, lmax=None, lmax_mask=None, nran=None, mask_nside=None):
    """Override module-level configuration and recompute derived globals."""
    global DZ, LMAX, LMAX_MASK, NRAN_FILES, MASK_NSIDE, Z_EDGES, ZKEYS
    if dz is not None:
        DZ = dz
    if lmax is not None:
        LMAX = lmax
    if lmax_mask is not None:
        LMAX_MASK = lmax_mask
    if nran is not None:
        NRAN_FILES = nran
    if mask_nside is not None:
        MASK_NSIDE = mask_nside
    Z_EDGES = np.round(np.arange(Z_MIN, Z_MAX + DZ / 2, DZ), 1)
    ZKEYS = [zkey(i) for i in range(len(Z_EDGES) - 1)]


def tag_suffix():
    """Identifier for the current configuration, used in cache/output names."""
    return f"dz{DZ}_lmax{LMAX}_lmask{LMAX_MASK}_nran{NRAN_FILES}"


# --------------------------------------------------------------------------- #
# stage 1 -- catalogs
# --------------------------------------------------------------------------- #

def _icrs_to_galactic(ra, dec):
    """Rotate ICRS -> Galactic in chunks, returning (l, b) in degrees."""
    l = np.empty(len(ra))
    b = np.empty(len(ra))
    for lo in range(0, len(ra), CHUNK):
        hi = min(lo + CHUNK, len(ra))
        c = SkyCoord(ra=ra[lo:hi] * u.degree,
                     dec=dec[lo:hi] * u.degree, frame="icrs").galactic
        l[lo:hi] = c.l.degree
        b[lo:hi] = c.b.degree
    return l, b


def _read_cat(path, columns):
    with fits.open(path, memmap=True) as h:
        return {c: np.asarray(h[1].data[c], dtype=np.float64) for c in columns}


def build_catalog_cache(force=False):
    """Read DESI catalogs, rotate to Galactic once, cache as .npz.

    The notebook re-ran SkyCoord inside the per-bin loop on 7.5M + 51.7M rows;
    doing it once up front is the single largest wall-clock saving here.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cols = ["Z", "RA", "DEC", "WEIGHT"]

    dat_path = CACHE / "cat_data.npz"
    if force or not dat_path.exists():
        t0 = time()
        d = _read_cat(DR2_FOLDER / "ELG_LOPnotqso_clustering.dat.fits", cols)
        l, b = _icrs_to_galactic(d["RA"], d["DEC"])
        np.savez(dat_path, l=l, b=b, z=d["Z"], w=d["WEIGHT"])
        print(f"[cats] data: {len(l):,} rows, sum(w)={d['WEIGHT'].sum():.4e}, "
              f"{time() - t0:.1f}s")
        del d, l, b

    ran_path = CACHE / "cat_rand.npz"
    if force or not ran_path.exists():
        t0 = time()
        parts = []
        for i in range(2):
            parts.append(_read_cat(
                DR2_FOLDER / f"ELG_LOPnotqso_{i}_clustering.ran.fits", cols))
        ra = np.concatenate([p["RA"] for p in parts])
        dec = np.concatenate([p["DEC"] for p in parts])
        z = np.concatenate([p["Z"] for p in parts])
        w = np.concatenate([p["WEIGHT"] for p in parts])
        # which random file each row came from, so runs can use a subset
        src = np.concatenate([np.full(len(p["RA"]), i, dtype=np.int8)
                              for i, p in enumerate(parts)])
        del parts
        l, b = _icrs_to_galactic(ra, dec)
        del ra, dec
        np.savez(ran_path, l=l, b=b, z=z, w=w, src=src)
        print(f"[cats] rand: {len(l):,} rows, sum(w)={w.sum():.4e}, "
              f"{time() - t0:.1f}s")


def load_tomo(which):
    """Return {zkey: dict(l, b, w)} for `which` in {'data', 'rand'}.

    Randoms are restricted to the first NRAN_FILES catalogs. Note that adding
    randoms mainly reduces the *variance* of the mask estimate: route A already
    removes the random shot-noise *bias* exactly, since NaMaster subtracts
    Nw = sum(w_rand^2)/(4pi) from the mask pseudo-C_ell when building the MCM.
    """
    f = np.load(CACHE / f"cat_{which}.npz")
    l, b, z, w = f["l"], f["b"], f["z"], f["w"]
    if which == "rand" and "src" in f.files:
        keep = f["src"] < NRAN_FILES
        l, b, z, w = l[keep], b[keep], z[keep], w[keep]
    out = {}
    for i in range(len(Z_EDGES) - 1):
        sel = (z >= Z_EDGES[i]) & (z < Z_EDGES[i + 1])
        out[zkey(i)] = {"l": l[sel], "b": b[sel], "w": w[sel]}
    return out



# --------------------------------------------------------------------------- #
# stage 2 -- CIB fields
# --------------------------------------------------------------------------- #

def build_cib_field(nu):
    """CIB NmtField with the footprint passed as a mask and Wl_eff as the beam.

    The notebook passed `np.ones_like(map_eff)` as the mask with the mask already
    baked into the map, so the CIB footprint was never deconvolved. NaMaster
    multiplies maps by the mask internally (masked_on_input defaults to False),
    so the *raw* NaN-cleaned map must be passed here.

    Wl_eff = pixfunc x Bl_eff (per the Lenz README, verified numerically), so
    passing it as `beam` deconvolves both the 5' beam and the HEALPix pixel
    window in one step. Do not additionally divide by hp.pixwin.
    """
    d = CIB_FOLDER / nu / NHI
    mask_bool = hp.read_map(d / "mask_bool.hpx.fits", dtype=bool)
    mask_apod = hp.read_map(d / "mask_apod.hpx.fits")
    map_fm = hp.read_map(d / "cib_fullmission.hpx.fits") * 1e6  # MJy/sr -> Jy/sr

    mask_cib = np.where(mask_bool, 1.0, 0.0) * mask_apod
    map_clean = np.where(mask_bool, map_fm, 0.0)  # NaN removal only

    wl = pd.read_csv(d / "windowfunctions.csv", comment="#")["Wl_eff"].values

    return nmt.NmtField(mask_cib, [map_clean], beam=wl[:LMAX + 1],
                        lmax=LMAX, lmax_mask=LMAX_MASK, n_iter=0)


# --------------------------------------------------------------------------- #
# stage 3 -- galaxy fields
# --------------------------------------------------------------------------- #

def _counts_map(cat, nside=NSIDE):
    pix = hp.ang2pix(nside, cat["l"], cat["b"], lonlat=True)
    return np.bincount(pix, weights=cat["w"],
                       minlength=hp.nside2npix(nside))


def build_completeness_mask(ran):
    """Weighted random density for ONE tomographic bin, estimated at MASK_NSIDE
    and upgraded to NSIDE.

    Must use the randoms of this bin, not the pooled catalog. Pooling all
    redshifts looks safe on footprint area (fsky at nside 256 is 0.3165 for all
    z vs 0.3155 for bin 0.8-1.0) but the completeness *pattern* is strongly
    redshift dependent: the normalised all-z and per-bin random densities
    correlate at only 0.68 at nside 512, with 55% of pixels differing by >20%.
    Using the pooled mask shifts the first bandpower by -3.7x. With per-bin
    randoms, routes B and C converge onto route A to ~1%.

    ud_grade with power=0 copies the coarse value into each child pixel, which
    is what we want: the mask represents an intensive quantity (expected
    weighted galaxy density per steradian), and only its shape matters since
    NaMaster sets alpha = sum(w_data) / mask_area internally.
    """
    return hp.ud_grade(_counts_map(ran, nside=MASK_NSIDE), NSIDE)


def build_gal_field(dat, ran, route, mask=None, retain_catalog=False):
    """Galaxy field for one tomographic bin. `route` in {'A', 'B', 'C'}.

    A vs B isolates the footprint representation (random catalog vs smooth
    map), B vs C isolates pixelising the data. `mask` is required for B and C.

    `retain_catalog` keeps the source positions on the field, which NaMaster
    (>=3.0) needs to build a covariance workspace from catalog fields.
    """
    pos_d = np.array([dat["l"], dat["b"]])

    if route == "A":
        pos_r = np.array([ran["l"], ran["b"]])
        return nmt.NmtFieldCatalogClustering(
            pos_d, dat["w"], pos_r, ran["w"],
            lmax=LMAX, lmax_mask=LMAX_MASK, lonlat=True,
            retain_catalog=retain_catalog)

    if mask is None:
        raise ValueError(f"route {route!r} needs a completeness mask")

    if route == "B":
        return nmt.NmtFieldCatalogClustering(
            pos_d, dat["w"], None, None, mask=mask,
            lmax=LMAX, lmax_mask=LMAX_MASK, lonlat=True)

    if route == "C":
        gal_counts = _counts_map(dat)
        good = mask > 0
        # normalise on the mask itself, so alpha matches NaMaster's convention
        alpha = gal_counts[good].sum() / mask[good].sum()
        delta_g = np.zeros(NPIX)
        delta_g[good] = gal_counts[good] / (alpha * mask[good]) - 1.0
        # beam=pixwin is what makes this comparable to A/B after decoupling:
        # the catalog routes carry no pixel window, this one does.
        return nmt.NmtField(mask, [delta_g], beam=hp.pixwin(NSIDE)[:LMAX + 1],
                            lmax=LMAX, lmax_mask=LMAX_MASK, n_iter=0)

    raise ValueError(f"unknown route {route!r}")


# --------------------------------------------------------------------------- #
# stage 4 -- workspaces and spectra
# --------------------------------------------------------------------------- #

def get_workspace(f1, f2, bins, tag):
    """Mode-coupling matrix, cached to disk.

    The MCM depends only on the two masks, and at lmax 2000 each one costs
    minutes. The notebook rebuilt all 12 inside the innermost loop.
    """
    path = CACHE / f"wsp_{tag}.fits"
    if path.exists():
        return nmt.NmtWorkspace.from_file(str(path))
    t0 = time()
    w = nmt.NmtWorkspace.from_fields(f1, f2, bins)
    w.write_to(str(path))
    print(f"  [mcm] {tag}: {time() - t0:.1f}s")
    return w


def measure(routes, zkeys, nus, force_wsp=False):
    bins = nmt.NmtBin.from_lmax_linear(LMAX, nlb=NLB)
    leff = bins.get_effective_ells()

    tomo_dat = load_tomo("data")
    tomo_ran = load_tomo("rand")

    need_mask = bool({"B", "C"} & set(routes))

    print(f"[cib] building {len(nus)} field(s)")
    cib_fields = {}
    for nu in nus:
        t0 = time()
        cib_fields[nu] = build_cib_field(nu)
        print(f"  [cib] {nu} GHz: {time() - t0:.1f}s")

    masks = {}
    if need_mask:
        for zk in zkeys:
            t0 = time()
            masks[zk] = build_completeness_mask(tomo_ran[zk])
            print(f"[mask] {zk} nside {MASK_NSIDE} -> {NSIDE}: "
                  f"fsky={(masks[zk] > 0).mean():.4f}, {time() - t0:.1f}s")

    results = {}
    for route in routes:
        for zk in zkeys:
            t0 = time()
            f_gal = build_gal_field(tomo_dat[zk], tomo_ran[zk], route,
                                    mask=masks.get(zk))
            print(f"[gal] route {route} {zk}: "
                  f"{len(tomo_dat[zk]['w']):,} gal, "
                  f"{len(tomo_ran[zk]['w']):,} ran, {time() - t0:.1f}s")

            for nu in nus:
                tag = (f"{route}_{zk}_{nu}_{tag_suffix()}"
                       + (f"_mns{MASK_NSIDE}" if route in "BC" else ""))
                if force_wsp:
                    (CACHE / f"wsp_{tag}.fits").unlink(missing_ok=True)
                w = get_workspace(f_gal, cib_fields[nu], bins, tag)
                pcl = nmt.compute_coupled_cell(f_gal, cib_fields[nu])
                cl = w.decouple_cell(pcl)[0]
                results[(route, zk, nu)] = cl
                print(f"  [cl] {tag}: cl[0]={cl[0]:.4e} cl[-1]={cl[-1]:.4e}")

            del f_gal

    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out = {"leff": leff, "lmax": LMAX, "lmax_mask": LMAX_MASK, "nhi": NHI,
           "dz": DZ, "nran": NRAN_FILES, "zkeys": np.array(ZKEYS)}
    for (route, zk, nu), cl in results.items():
        out[f"cl_{route}_{zk}_{nu}"] = cl
    path = OUT_FOLDER / f"cl_galxcib_{NHI}_{tag_suffix()}.npz"
    np.savez(path, **out)
    print(f"[out] wrote {path}")
    return results, leff


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["cats", "measure", "all"], default="all")
    p.add_argument("--routes", nargs="+", default=["A", "C"], choices=["A", "B", "C"])
    p.add_argument("--smoke", action="store_true",
                   help="one (zbin, nu) pair only, for timing")
    p.add_argument("--dz", type=float, default=None,
                   help="tomographic bin width (0.2, 0.4, 0.8)")
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--lmax-mask", type=int, default=None)
    p.add_argument("--nran", type=int, default=None,
                   help="number of random catalogs to use")
    p.add_argument("--mask-nside", type=int, default=None)
    p.add_argument("--force-cats", action="store_true")
    p.add_argument("--force-wsp", action="store_true")
    args = p.parse_args()
    configure(dz=args.dz, lmax=args.lmax, lmax_mask=args.lmax_mask,
              nran=args.nran, mask_nside=args.mask_nside)
    print(f"[cfg] {tag_suffix()} mask_nside={MASK_NSIDE} bins={ZKEYS}")

    if args.stage in ("cats", "all"):
        build_catalog_cache(force=args.force_cats)

    if args.stage in ("measure", "all"):
        zkeys = ZKEYS[:1] if args.smoke else ZKEYS
        nus = NULIST[:1] if args.smoke else NULIST
        t0 = time()
        measure(args.routes, zkeys, nus, force_wsp=args.force_wsp)
        print(f"[done] {time() - t0:.1f}s")


if __name__ == "__main__":
    main()
