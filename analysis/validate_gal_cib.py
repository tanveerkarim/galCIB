"""
Validation plots and convergence tests for the DESI ELG x Lenz CIB measurement
produced by analysis/measure_gal_cib.py.

The point of the A/B/C comparison is to settle whether the catalog-based scheme
(NmtFieldCatalogClustering) can be used. Compare *decoupled* C_ells only --
coupled pseudo-C_ells from catalog fields carry an unnormalised mask scale of
sum(w_rand) ~ 1.3e8 and are ~1e16 above their map-based counterparts by
construction.

The cross-spectrum is the right thing to compare: unlike the auto, gal x CIB
carries no shot-noise term, since galaxy Poisson noise is uncorrelated with the
CIB map.

Usage
-----
    python analysis/validate_gal_cib.py --compare     # A vs B vs C panel
    python analysis/validate_gal_cib.py --lmax-mask-test  # lmax_mask convergence
    python analysis/validate_gal_cib.py --null        # rotated-catalog null test
"""

import argparse
from pathlib import Path
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pandas as pd
import pymaster as nmt

import measure_gal_cib as m


OUT = m.OUT_FOLDER


def _load():
    path = OUT / f"cl_galxcib_{m.NHI}_{m.tag_suffix()}.npz"
    return np.load(path)


def compare():
    """A vs B vs C decoupled cross-spectra on a bin x frequency grid.

    Rows are tomographic bins, columns are frequencies. The top half of each
    cell is l*Cl, the bottom half the fractional difference from route A.
    """
    f = _load()
    leff = f["leff"]
    use = leff < m.ELL_USE_MAX
    routes = sorted({k.split("_")[1] for k in f.files if k.startswith("cl_")})

    nr, nc = len(m.ZKEYS), len(m.NULIST)
    fig, axes = plt.subplots(2 * nr, nc, figsize=(4.2 * nc, 5.0 * nr),
                             squeeze=False, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1] * nr})
    for i, zk in enumerate(m.ZKEYS):
        for j, nu in enumerate(m.NULIST):
            ax, axr = axes[2 * i, j], axes[2 * i + 1, j]
            ref = f[f"cl_A_{zk}_{nu}"]
            for route in routes:
                cl = f[f"cl_{route}_{zk}_{nu}"]
                ax.plot(leff[use], (leff * cl)[use], marker="o", ms=3,
                        label=f"route {route}")
                if route != "A":
                    axr.plot(leff[use], (cl / ref - 1)[use], marker="o", ms=3)
            ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"z {zk}   {nu} GHz", fontsize=10)
            if j == 0:
                ax.set_ylabel(r"$\ell\,C_\ell^{g\times\rm CIB}$ [Jy/sr]")
                axr.set_ylabel("ratio to A $-$ 1")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            axr.axhline(0, color="k", lw=0.5)
            axr.set_ylim(-0.15, 0.15)
            axr.set_xscale("log")
            if i == nr - 1:
                axr.set_xlabel(r"$\ell$")

            for route in routes:
                if route == "A":
                    continue
                sel = (leff > 100) & use
                dev = np.mean(np.abs(f[f"cl_{route}_{zk}_{nu}"][sel] / ref[sel] - 1))
                print(f"  {zk} {nu} route {route}: mean |ratio-1| "
                      f"(100<l<{m.ELL_USE_MAX}) = {dev:.4f}")

    fig.tight_layout()
    path = OUT / "validate_routes.png"
    fig.savefig(path, dpi=110)
    print(f"[out] wrote {path}")



def lmax_mask_convergence(zk=None, nu="545", values=(3000, 4000, 5000)):
    """NaMaster's docs warn to check lmax_mask sensitivity for catalog fields."""
    zk = zk or m.ZKEYS[0]
    dat = m.load_tomo("data")[zk]
    ran = m.load_tomo("rand")[zk]

    orig = m.LMAX_MASK
    out = {}
    for lm in values:
        m.LMAX_MASK = lm
        t0 = time()
        f_gal = m.build_gal_field(dat, ran, "A")
        f_cib = m.build_cib_field(nu)
        bins = nmt.NmtBin.from_lmax_linear(m.LMAX, nlb=m.NLB)
        w = nmt.NmtWorkspace.from_fields(f_gal, f_cib, bins)
        out[lm] = w.decouple_cell(nmt.compute_coupled_cell(f_gal, f_cib))[0]
        leff = bins.get_effective_ells()
        print(f"[lmax_mask={lm}] {time() - t0:.1f}s")
        del f_gal, f_cib, w
    m.LMAX_MASK = orig

    ref = out[values[-1]]
    for lm in values[:-1]:
        sel = (leff > 100) & (leff < m.ELL_USE_MAX)
        print(f"  lmax_mask={lm} vs {values[-1]}: mean |ratio-1| = "
              f"{np.mean(np.abs(out[lm][sel] / ref[sel] - 1)):.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for lm, cl in out.items():
        ax.plot(leff, cl / ref - 1, marker="o", ms=3, label=f"lmax_mask={lm}")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(f"ratio to lmax_mask={values[-1]} $-$ 1")
    ax.legend()
    fig.tight_layout()
    path = OUT / "validate_lmax_mask.png"
    fig.savefig(path, dpi=130)
    print(f"[out] wrote {path}")


def null_test(zk=None, nu="545", shift_deg=120.0):
    """Rotate galaxies AND randoms together by a large offset in galactic
    longitude; the cross with the CIB map should be consistent with zero.

    Both catalogs must be rotated by the same amount. Rotating only the data
    leaves the galaxies sitting outside their own random footprint, so the
    overdensity field is meaningless rather than merely decorrelated -- that
    produces a huge spurious first bandpower, not a null.
    """
    zk = zk or m.ZKEYS[0]
    dat = dict(m.load_tomo("data")[zk])
    ran = dict(m.load_tomo("rand")[zk])
    dat["l"] = (dat["l"] + shift_deg) % 360.0
    ran["l"] = (ran["l"] + shift_deg) % 360.0

    f_gal = m.build_gal_field(dat, ran, "A")
    f_cib = m.build_cib_field(nu)
    bins = nmt.NmtBin.from_lmax_linear(m.LMAX, nlb=m.NLB)
    leff = bins.get_effective_ells()
    w = nmt.NmtWorkspace.from_fields(f_gal, f_cib, bins)
    cl_null = w.decouple_cell(nmt.compute_coupled_cell(f_gal, f_cib))[0]

    sig = _load().get(f"cl_A_{zk}_{nu}")
    print(f"[null] shift={shift_deg} deg, {zk} {nu} GHz")
    if sig is not None:
        sel = (leff > 100) & (leff < m.ELL_USE_MAX)
        print(f"  |null|/|signal| median (100<l<ELL_USE_MAX) = "
              f"{np.median(np.abs(cl_null[sel] / sig[sel])):.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(leff, leff * cl_null, marker="o", ms=3, label="null (rotated)")
    if sig is not None:
        ax.plot(leff, leff * sig, marker="o", ms=3, label="signal")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell\,C_\ell$")
    ax.legend()
    fig.tight_layout()
    path = OUT / "validate_null.png"
    fig.savefig(path, dpi=130)
    print(f"[out] wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--lmax-mask-test", action="store_true")
    p.add_argument("--null", action="store_true")
    p.add_argument("--snr", action="store_true")
    p.add_argument("--snr-plot", action="store_true")
    p.add_argument("--fsky", action="store_true")
    p.add_argument("--shotnoise", action="store_true")
    p.add_argument("--spectra", action="store_true")
    p.add_argument("--jackknife", action="store_true")
    p.add_argument("--nside-jk", type=int, default=4)
    p.add_argument("--ell-min", type=float, default=None)
    p.add_argument("--dz", type=float, default=None)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--lmax-mask", type=int, default=None)
    p.add_argument("--nran", type=int, default=None)
    p.add_argument("--nu", default="545")
    p.add_argument("--zbin", default=None)
    a = p.parse_args()
    m.configure(dz=a.dz, lmax=a.lmax, lmax_mask=a.lmax_mask, nran=a.nran)
    print(f"[cfg] {m.tag_suffix()} bins={m.ZKEYS}")

    if a.compare:
        compare()
    if a.lmax_mask_test:
        lmax_mask_convergence(zk=a.zbin, nu=a.nu)
    if a.null:
        null_test(zk=a.zbin, nu=a.nu)
    if a.snr:
        snr()
    if a.snr_plot:
        snr_plot(ell_min=a.ell_min)
    if a.fsky:
        fsky_report()
    if a.shotnoise:
        shotnoise()
    if a.spectra:
        spectra_plot(dz=a.dz or 0.2)
    if a.jackknife:
        jackknife(zk=a.zbin, nu=a.nu, nside_jk=a.nside_jk)
    if not (a.compare or a.lmax_mask_test or a.null or a.snr or a.snr_plot or a.fsky or a.shotnoise or a.spectra or a.jackknife):
        p.print_help()



def snr(nus=None, zkeys=None):
    """Gaussian covariance and cumulative S/N per (bin, frequency).

    Uses NaMaster >= 3.0's catalog covariance path, so the covariance is built
    from the route-A field itself rather than a map-based proxy:
    `retain_catalog=True` keeps the source positions, NmtCovarianceWorkspace
    .from_fields accepts catalog fields, and get_iNKA_cell supplies the
    improved-narrow-kernel-approximation input spectra measured from the data.
    """
    nus = nus or m.NULIST
    zkeys = zkeys or m.ZKEYS
    bins = nmt.NmtBin.from_lmax_linear(m.LMAX, nlb=m.NLB)
    leff = bins.get_effective_ells()
    meas = _load()

    tomo_d, tomo_r = m.load_tomo("data"), m.load_tomo("rand")
    out = {"leff": leff}
    for zk in zkeys:
        fg = m.build_gal_field(tomo_d[zk], tomo_r[zk], "A", retain_catalog=True)
        for nu in nus:
            t0 = time()
            fc = m.build_cib_field(nu)
            w_gi = nmt.NmtWorkspace.from_fields(fg, fc, bins)
            cw = nmt.NmtCovarianceWorkspace.from_fields(fg, fc, fg, fc)

            # iNKA effective spectra, measured from the data themselves
            cl_gg = nmt.get_iNKA_cell(fg, fg)
            cl_gi = nmt.get_iNKA_cell(fg, fc)
            cl_ii = nmt.get_iNKA_cell(fc, fc)
            cov = cw.gaussian_covariance(cl_gg, cl_gi, cl_gi, cl_ii,
                                         wa=w_gi, wb=w_gi)

            cl = meas[f"cl_A_{zk}_{nu}"]
            cum = np.array([
                np.sqrt(max(cl[:n] @ np.linalg.solve(cov[:n, :n], cl[:n]), 0))
                for n in range(1, len(cl) + 1)])
            out[f"cov_{zk}_{nu}"] = cov
            out[f"cum_{zk}_{nu}"] = cum
            out[f"cl_{zk}_{nu}"] = cl
            n600 = int(np.sum(leff < m.ELL_USE_MAX))
            print(f"[snr] {zk} {nu}: S/N(l<{m.ELL_USE_MAX})={cum[n600-1]:.1f}  "
                  f"total={cum[-1]:.1f}  ({time() - t0:.0f}s)", flush=True)
            del cw, cov, fc, w_gi
        del fg

    np.savez(OUT / f"snr_{m.tag_suffix()}.npz", **out)
    print(f'[out] wrote snr_{m.tag_suffix()}.npz')


def snr_plot(smooth=5, ell_min=None):
    """Cumulative S/N vs ell and its derivative, per z-bin and CIB frequency.

    With `ell_min` set, accumulation starts at the first bandpower above the cut
    and the S/N is recomputed from the *restricted* covariance block,
    cl[i0:n] . cov[i0:n, i0:n]^-1 . cl[i0:n] -- not rescaled from the full-range
    result, since dropping bandpowers changes the inverse.

    Right column is d(S/N)/dl = diff(cum)/diff(l). The raw curve spikes downward
    wherever a single bandpower contributes nothing, so a running median is
    overlaid; the dashed line marks 10% of each curve's peak gain.
    """
    f = np.load(OUT / f"snr_{m.tag_suffix()}.npz")
    leff = f["leff"]
    zks = [zk for zk in m.ZKEYS if f"cum_{zk}_{m.NULIST[0]}" in f.files]
    if not zks:
        raise SystemExit("no cumulative S/N -- run --snr first")
    i0 = 0 if ell_min is None else int(np.searchsorted(leff, ell_min))
    lcut = leff[i0:]
    lmid = 0.5 * (lcut[1:] + lcut[:-1])
    colors = dict(zip(m.NULIST, ["C0", "C1", "C2"]))

    def runmed(y, n):
        pad = n // 2
        yp = np.pad(y, pad, mode="edge")
        return np.array([np.median(yp[i:i + n]) for i in range(len(y))])

    fig, axes = plt.subplots(len(zks), 2, figsize=(12, 3.3 * len(zks)),
                             squeeze=False, sharex=True)
    hdr = f" (l_min = {ell_min})" if ell_min else ""
    print(f"{'bin':>10} {'nu':>5} {'total':>7} {'l(90%)':>7} {'l(95%)':>7} "
          f"{'l(gain<10%pk)':>14} {'lost to cut':>12}")
    for i, zk in enumerate(zks):
        axc, axd = axes[i, 0], axes[i, 1]
        for nu in m.NULIST:
            cov = f[f"cov_{zk}_{nu}"]
            cl = f[f"cl_{zk}_{nu}"]
            cum = np.array([
                np.sqrt(max(cl[i0:n] @ np.linalg.solve(cov[i0:n, i0:n],
                                                       cl[i0:n]), 0))
                for n in range(i0 + 1, len(cl) + 1)])
            tot = cum[-1]
            full = f[f"cum_{zk}_{nu}"][-1]
            axc.plot(lcut, cum, marker="o", ms=3, color=colors[nu],
                     label=f"{nu} GHz")
            d = np.diff(cum) / np.diff(lcut)
            dm = runmed(d, smooth)
            axd.plot(lmid, d, color=colors[nu], lw=0.7, alpha=0.30)
            axd.plot(lmid, dm, color=colors[nu], lw=1.8)
            axd.axhline(0.10 * dm.max(), color=colors[nu], ls="--", lw=0.7)
            below = np.where(dm < 0.10 * dm.max())[0]
            lflat = lmid[below[0]] if len(below) else np.nan
            l90 = lcut[np.searchsorted(cum, 0.90 * tot)]
            l95 = lcut[np.searchsorted(cum, 0.95 * tot)]
            print(f"{zk:>10} {nu:>5} {tot:7.1f} {l90:7.0f} {l95:7.0f} "
                  f"{lflat:14.0f} {1 - tot / full:11.1%}")
        for a in (axc, axd):
            a.axvline(m.ELL_USE_MAX, color="k", ls=":", lw=1)
            if ell_min:
                a.axvspan(0, ell_min, color="grey", alpha=0.18, lw=0)
            a.set_xlim(0, leff[-1] + m.NLB)
        axc.set_ylim(0, None)
        axc.set_ylabel(f"z {zk}\ncumulative S/N")
        axd.set_ylabel(r"$\Delta(S/N)/\Delta\ell$")
        axd.set_yscale("log")
        if i == 0:
            axc.legend(fontsize=8, loc="lower right")
            axc.set_title(f"cumulative S/N{hdr}")
            axd.set_title(r"$\Delta(S/N)/\Delta\ell$  "
                          "(bold = running median, dashed = 10% of peak)")
        if i == len(zks) - 1:
            axc.set_xlabel(r"$\ell$")
            axd.set_xlabel(r"$\ell$")

    fig.tight_layout()
    sfx = f"_lmin{ell_min}" if ell_min else ""
    path = OUT / f"validate_snr_{m.tag_suffix()}{sfx}.png"
    fig.savefig(path, dpi=130)
    print(f"[out] wrote {path}")



def fsky_report():
    """Sky fractions and bandpower correlations, to judge a trustworthy l_min.

    Reports the standard w-moment sky fractions for each galaxy x CIB pair, the
    number of modes in the first bandpower, and the adjacent-bandpower
    correlation from the Gaussian covariance. The last is the empirical test:
    if neighbouring bandpowers are strongly correlated, the mode-coupling has
    not been cleanly deconvolved at that scale and l_min should be raised.
    """
    bins = nmt.NmtBin.from_lmax_linear(m.LMAX, nlb=m.NLB)
    leff = bins.get_effective_ells()
    tomo_r = m.load_tomo("rand")
    snr_path = OUT / f"snr_{m.tag_suffix()}.npz"
    snr = np.load(snr_path) if snr_path.exists() else None

    cib_masks = {}
    for nu in m.NULIST:
        d = m.CIB_FOLDER / nu / m.NHI
        mb = hp.read_map(d / "mask_bool.hpx.fits", dtype=bool)
        ma = hp.read_map(d / "mask_apod.hpx.fits")
        cib_masks[nu] = np.where(mb, 1.0, 0.0) * ma

    print(f"{'bin':>10} {'nu':>4} {'fsky_g':>7} {'fsky_I':>7} {'fsky_x':>7} "
          f"{'fsky_eff':>8} {'Nmode(bp0)':>11} {'corr(bp0,bp1)':>13}")
    for zk in m.ZKEYS:
        wg = m.build_completeness_mask(tomo_r[zk])
        wg = wg / wg.max()
        fsky_g = np.mean(wg > 0)
        for nu in m.NULIST:
            wi = cib_masks[nu]
            prod = wg * wi
            fsky_i = np.mean(wi)
            fsky_x = np.mean(prod)
            # effective sky fraction for mode counting: <w1 w2>^2 / <w1^2 w2^2>
            fsky_eff = np.mean(prod) ** 2 / np.mean(prod ** 2)
            nmode = (2 * leff[0] + 1) * m.NLB * fsky_eff
            c = ""
            if snr is not None and f"cov_{zk}_{nu}" in snr.files:
                cov = snr[f"cov_{zk}_{nu}"]
                dg = np.sqrt(np.diag(cov))
                c = f"{cov[0, 1] / (dg[0] * dg[1]):13.3f}"
            print(f"{zk:>10} {nu:>4} {fsky_g:7.4f} {fsky_i:7.4f} {fsky_x:7.4f} "
                  f"{fsky_eff:8.4f} {nmode:11.0f} {c:>13}")

    if snr is not None:
        print("\nadjacent-bandpower correlation vs l (bin 0.8-1.0, 545 GHz):")
        cov = snr[f"cov_{m.ZKEYS[0]}_545"]
        dg = np.sqrt(np.diag(cov))
        for i in range(6):
            print(f"  l={leff[i]:6.1f}  corr(bp{i},bp{i+1}) = "
                  f"{cov[i, i+1] / (dg[i] * dg[i+1]):+.3f}")



def shotnoise(dzs=(0.2, 0.4, 0.8), nu="545"):
    """Galaxy shot noise vs clustering signal, and its effect on gal x CIB.

    The cross-spectrum has NO shot-noise term in the mean -- galaxy Poisson
    noise is uncorrelated with the CIB map. Shot noise enters only the
    *variance*, through the C_l^gg factor in the Gaussian covariance:

        Var(C_l^gI) ~ [ (C_l^gI)^2 + C_l^gg,tot * C_l^II,tot ] / ((2l+1) dl fsky)

    so it hurts as sqrt(N_shot), not linearly, and it never biases the signal.
    N_shot is measured directly here: NaMaster subtracts Nf when the same field
    object is passed twice, so the difference between the two-object and
    one-object autos is exactly the shot-noise term.
    """
    orig_dz = m.DZ
    fig, axes = plt.subplots(len(dzs), 2, figsize=(11, 3.4 * len(dzs)),
                             squeeze=False)
    print(f"{'dz':>5} {'bin':>10} {'Ngal':>10} {'nbar[/sr]':>11} "
          f"{'N_shot':>10} {'1/nbar':>10} {'l_cross':>8}")
    for i, dz in enumerate(dzs):
        m.configure(dz=dz)
        bins = nmt.NmtBin.from_lmax_linear(m.LMAX, nlb=m.NLB)
        leff = bins.get_effective_ells()
        td, tr = m.load_tomo("data"), m.load_tomo("rand")
        meas = _load()
        axg, axc = axes[i, 0], axes[i, 1]
        for j, zk in enumerate(m.ZKEYS):
            fg = m.build_gal_field(td[zk], tr[zk], "A")
            fg2 = m.build_gal_field(td[zk], tr[zk], "A")
            w = nmt.NmtWorkspace.from_fields(fg, fg, bins)
            tot = w.decouple_cell(nmt.compute_coupled_cell(fg, fg2))[0]
            sig = w.decouple_cell(nmt.compute_coupled_cell(fg, fg))[0]
            nsh = np.median(tot - sig)

            wts = td[zk]["w"]
            mask = m.build_completeness_mask(tr[zk])
            fsky = np.mean(mask > 0)
            nbar = wts.sum() ** 2 / np.sum(wts ** 2) / (4 * np.pi * fsky)
            cross = leff[sig < nsh]
            lx = cross[0] if len(cross) else np.nan
            print(f"{dz:>5} {zk:>10} {len(wts):10,d} {nbar:11.3e} "
                  f"{nsh:10.3e} {1/nbar:10.3e} {lx:8.0f}")

            c = f"C{j}"
            axg.loglog(leff, np.abs(sig), color=c, lw=1.4, label=f"z {zk}")
            axg.axhline(nsh, color=c, ls="--", lw=1.0)
            cl = meas[f"cl_A_{zk}_{nu}"]
            axc.loglog(leff, np.abs(cl), color=c, lw=1.4, label=f"z {zk}")
            axc.loglog(leff, np.where(cl < 0, np.abs(cl), np.nan), color=c,
                       ls="", marker="v", ms=4)
            del fg, fg2, w
        axg.set_title(f"dz={dz}: galaxy auto (solid) vs shot noise (dashed)",
                      fontsize=9)
        axg.set_ylabel(r"$C_\ell^{gg}$")
        # log axis shows |C_l|, so flag any negative bandpowers explicitly
        axc.set_title(f"dz={dz}: measured gal x CIB cross-spectrum at {nu} GHz"
                      "\n(triangles mark negative bandpowers, if any)",
                      fontsize=9)
        axc.set_ylabel(r"$|C_\ell^{g \times \rm CIB}|$ [Jy/sr]")
        for a in (axg, axc):
            a.set_xlabel(r"$\ell$")
            a.legend(fontsize=7)
            a.axvline(m.ELL_USE_MAX, color="k", ls=":", lw=1)
    m.configure(dz=orig_dz)
    fig.tight_layout()
    path = OUT / "validate_shotnoise.png"
    fig.savefig(path, dpi=130)
    print(f"[out] wrote {path}")



def spectra_plot(dz=0.2, ell_max=None):
    """Standalone C_l vs l for the measured gal x CIB cross-spectra.

    One panel per CIB frequency, all tomographic bins overlaid, with error bars
    from the diagonal of the Gaussian covariance. Log axes show |C_l|, so any
    negative bandpower is drawn as an open symbol.
    """
    m.configure(dz=dz)
    meas = _load()
    leff = meas["leff"]
    spath = OUT / f"snr_{m.tag_suffix()}.npz"
    cov = np.load(spath) if spath.exists() else None
    ell_max = ell_max or leff[-1]
    keep = leff <= ell_max

    fig, axes = plt.subplots(1, len(m.NULIST), figsize=(5.0 * len(m.NULIST), 4.6),
                             sharey=True)
    for j, nu in enumerate(m.NULIST):
        ax = axes[j]
        for k, zk in enumerate(m.ZKEYS):
            cl = meas[f"cl_A_{zk}_{nu}"]
            err = None
            if cov is not None and f"cov_{zk}_{nu}" in cov.files:
                err = np.sqrt(np.diag(cov[f"cov_{zk}_{nu}"]))
            c = f"C{k}"
            pos = keep & (cl > 0)
            neg = keep & (cl <= 0)
            ax.errorbar(leff[pos], cl[pos],
                        yerr=None if err is None else err[pos],
                        color=c, marker="o", ms=4, lw=1.2, capsize=2,
                        label=f"z {zk}")
            if neg.any():
                ax.errorbar(leff[neg], -cl[neg],
                            yerr=None if err is None else err[neg],
                            color=c, marker="o", ms=5, mfc="white", ls="",
                            capsize=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_title(f"{nu} GHz")
        ax.axvline(m.ELL_USE_MAX, color="k", ls=":", lw=1)
        # clip to the signal range: without this the near-zero high-l points
        # stretch the log axis over 5 decades and squash the actual signal
        lo = np.percentile([meas[f"cl_A_{z}_{nu}"][keep].max()
                            for z in m.ZKEYS], 0) / 300.0
        hi = max(meas[f"cl_A_{z}_{nu}"][keep].max() for z in m.ZKEYS) * 3
        ax.set_ylim(lo, hi)
        if j == 0:
            ax.set_ylabel(r"$C_\ell^{g \times \rm CIB}$  [Jy/sr]")
            ax.legend(fontsize=8)
    fig.suptitle(f"DESI DR2 ELG x Lenz CIB, dz={dz} "
                 "(open symbols = negative bandpowers)", fontsize=10)
    fig.tight_layout()
    path = OUT / f"spectra_dz{dz}.png"
    fig.savefig(path, dpi=140)
    print(f"[out] wrote {path}")



def jackknife(zk=None, nu="545", nside_jk=4, min_area_frac=0.25):
    """Delete-one-patch jackknife covariance, compared with the Gaussian one.

    The Gaussian covariance from get_iNKA_cell/gaussian_covariance omits the
    trispectrum, which is exactly what the 1-halo regime contributes at high l,
    so its high-l errors may be optimistic. A jackknife makes no Gaussianity
    assumption and tests that directly.

    Each sample deletes one HEALPix superpixel from the galaxies, the randoms
    AND the CIB mask, then recomputes the mode-coupling matrix -- removing a
    patch changes the mask, so the MCM cannot be reused.

    Caveat: a jackknife cannot probe scales larger than a patch. At nside_jk=4
    the patches are ~215 deg^2 (~15 deg across), so results below l ~ 180/15
    = 12, and realistically the first bandpower or two, are not trustworthy.
    """
    zk = zk or m.ZKEYS[0]
    bins = nmt.NmtBin.from_lmax_linear(m.LMAX, nlb=m.NLB)
    leff = bins.get_effective_ells()
    dat = m.load_tomo("data")[zk]
    ran = m.load_tomo("rand")[zk]

    d = m.CIB_FOLDER / nu / m.NHI
    mask_bool = hp.read_map(d / "mask_bool.hpx.fits", dtype=bool)
    mask_apod = hp.read_map(d / "mask_apod.hpx.fits")
    cib_map = np.where(mask_bool, hp.read_map(d / "cib_fullmission.hpx.fits"), 0.) * 1e6
    cib_mask = np.where(mask_bool, 1.0, 0.0) * mask_apod
    wl = pd.read_csv(d / "windowfunctions.csv", comment="#")["Wl_eff"].values

    # superpixel id for every map pixel, and for every source
    theta, phi = hp.pix2ang(m.NSIDE, np.arange(m.NPIX))
    sp_map = hp.ang2pix(nside_jk, theta, phi)
    sp_dat = hp.ang2pix(nside_jk, dat["l"], dat["b"], lonlat=True)
    sp_ran = hp.ang2pix(nside_jk, ran["l"], ran["b"], lonlat=True)

    # keep patches that carry a decent share of the overlap
    area = np.bincount(sp_map, weights=cib_mask * (m.build_completeness_mask(ran) > 0),
                       minlength=hp.nside2npix(nside_jk))
    typical = np.median(area[area > 0])
    patches = np.where(area > min_area_frac * typical)[0]
    print(f"[jk] {zk} {nu} GHz: {len(patches)} patches at nside_jk={nside_jk} "
          f"({hp.nside2pixarea(nside_jk, degrees=True):.0f} deg^2 each), "
          f"{len(leff)} bandpowers")

    cls = []
    for i, pk in enumerate(patches):
        t0 = time()
        kd, kr = sp_dat != pk, sp_ran != pk
        sub_d = {k: v[kd] for k, v in dat.items()}
        sub_r = {k: v[kr] for k, v in ran.items()}
        cm = np.where(sp_map == pk, 0.0, cib_mask)
        fc = nmt.NmtField(cm, [cib_map], beam=wl[:m.LMAX + 1],
                          lmax=m.LMAX, lmax_mask=m.LMAX_MASK, n_iter=0)
        fg = m.build_gal_field(sub_d, sub_r, "A")
        w = nmt.NmtWorkspace.from_fields(fg, fc, bins)
        cls.append(w.decouple_cell(nmt.compute_coupled_cell(fg, fc))[0])
        if i % 5 == 0:
            print(f"  [jk] {i + 1}/{len(patches)}  ({time() - t0:.0f}s)", flush=True)
        del fg, fc, w

    cls = np.array(cls)
    n = len(cls)
    resid = cls - cls.mean(axis=0)
    cov_jk = (n - 1) / n * resid.T @ resid
    np.savez(OUT / f"jackknife_{zk}_{nu}_{m.tag_suffix()}.npz",
             leff=leff, cls=cls, cov=cov_jk, patches=patches)

    sg = None
    spath = OUT / f"snr_{m.tag_suffix()}.npz"
    if spath.exists():
        s = np.load(spath)
        if f"cov_{zk}_{nu}" in s.files:
            sg = np.sqrt(np.diag(s[f"cov_{zk}_{nu}"]))
    ejk = np.sqrt(np.diag(cov_jk))
    print(f"\n{'l':>7} {'sigma_JK':>11} {'sigma_G':>11} {'JK/Gauss':>9}")
    for i in range(len(leff)):
        g = f"{sg[i]:11.3e}" if sg is not None else f"{'--':>11}"
        r = f"{ejk[i]/sg[i]:9.2f}" if sg is not None else f"{'--':>9}"
        if i % 3 == 0 or i == len(leff) - 1:
            print(f"{leff[i]:7.1f} {ejk[i]:11.3e} {g} {r}")
    if sg is not None:
        for lo, hi in [(100, 600), (600, 1200), (1200, 2000)]:
            k = (leff > lo) & (leff < hi)
            print(f"  median JK/Gauss over {lo}<l<{hi}: "
                  f"{np.median(ejk[k]/sg[k]):.2f}")
    return cov_jk



if __name__ == "__main__":
    main()
