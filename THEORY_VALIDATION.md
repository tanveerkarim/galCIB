# galCIB halo model — validation log

Running record of validating `src/galCIB/` against the reference implementation
`../DopplerCIB` and against the Y23 paper (arXiv:2310.10848).

**Ground truth = DopplerCIB** for everything it implements. The paper arbitrates
only where DopplerCIB has no counterpart (i.e. the Y23 parametric SED — its
C_ℓ pipeline only ever uses the tabulated SED, `CIB_halo.py:34`).

Companion document: `SYSTEMATICS.md` (measurement side).

---

## Status

| area | state |
|---|---|
| Halo structure layer (HMF, bias, HOD, SED) | **verified bit-exact** |
| **CIB×CIB vs DopplerCIB** | **all 21 pairs agree to 0.4% median** — see §3a |
| **gal×CIB vs DopplerCIB** | **all 3 frequencies agree to 0.27%**, ℓ-flat — see §3b |
| Residual 0.27% | fully accounted for: colossus vs astropy Ω_m — see §3b |
| Y23 SED path | **2 bugs found and fixed** |
| Environment | `colossus`, `camb`, `pytest`, editable `galCIB` in the `cib` conda env |

---

## 1. Verified bit-exact against DopplerCIB

Driven from identical inputs (shared z, k, M grids; reference P(k) injected).
All ratios galCIB/DopplerCIB:

| component | max deviation | reference |
|---|---|---|
| HMF `dn/dlog₁₀M` (incl. `ln10`, h-units) | **0** | `data/hmfz_h_DopplerCIB.p` |
| Halo bias (hand-coded T10 vs colossus `tinker10`) | **0** | `hmf_unfw_bias.py:381` |
| M21 tabulated SED | **0** | `input_var_cibmean.py:40-46` |
| Galaxy HOD `Ncen` (mHMQ), `Nsat` | **0** | `Gal_halo.py` DESI_ELG branch |
| `Ncen_IR` driven to 1 for apples-to-apples | **0** | n/a |

Hand-coding the Tinker10 bias is a design choice, but it must still *agree*
numerically with colossus — it does, exactly.

## 2. Inputs — two traps

**P(k):** galCIB's internally computed P(k) differs from the pickle DopplerCIB
was fed by up to **3.6%, k-dependently** (0.965 at k=1e-4 → 1.006 at k=10). This
is camb-version drift, not a model bug, but it must be controlled or it
contaminates every 2-halo comparison. `test/reference/harness.py` injects the
reference P(k) into `Cosmology.pk_grid`; that propagates consistently because
`sigma(R)` → `nu` → bias and `c(M,z)` all read it.

**n(z):** DopplerCIB does **not** use `data_files/dndz_DESI_ELG.txt`.
`Gal_halo.py:113` carries a live `#FIXME: temporary test` that reads
`tmp/pz.npz` instead — that file is the shared n(z), and it is what §3b drives
both codes with. (The commented-out lines just below it read `zrange` /
`dndz.mean(axis=0)`, the structure of `data/gal/dndz_extended.p`; that was the
n(z) behind the retracted 2024-08 comparison, not the current one.)

Also note `compute_Wg` does **not** normalise p(z) (`survey/window.py`), whereas
DopplerCIB divides by `N = simpson(dn/dz, z)` (`Gal_halo.py:211`). Callers must
supply a normalised p(z).

## 3a. CIB×CIB — PASSES to sub-percent

Reference `data/cl_cib_dopplercib_planck.npz` (2025-11-11), the only saved
output postdating the shared P(k)/HMF. Cleanest comparison available: no galaxy
side at all, and the colour-correction conventions agree (DopplerCIB applies
`fc*cc` to the CIB auto at `CIB_halo.py:200`, galCIB applies `cc_II`).

Ratio galCIB/DopplerCIB, median over 200<ℓ<1500:

| pair | no shot noise | + shot noise |
|---|---|---|
| 353×353 | 0.813 | **0.9972** |
| 545×545 | 0.868 | **0.9997** |
| 857×857 | 0.893 | **1.0122** |
| 353×545 | 0.998 | 0.9980 |
| 353×857 | 1.003 | 1.0028 |
| 545×857 | 1.005 | 1.0053 |

**All 21 frequency pairs: min 0.9964, median 1.0005, max 1.0122.**

The diagnosis came straight from the pattern: off-diagonals matched to 0.2–0.5%
while diagonals were deficient with a deficit growing in ℓ — the signature of a
flat, **diagonal-only** shot-noise term. DopplerCIB adds exactly that
(`CIB_halo.py:250-281`, zero cross-frequency, values from the bestfit file rows
4:8, not colour-corrected). galCIB had been run with `theta_sn_II = 0`.

That the *off-diagonal* pairs — which carry no shot noise — already agreed to
0.2–0.5% before any shot term was added is the strongest single check in this
document: it exercises emissivity, SED, HMF, bias, NFW profile and Limber
projection end to end with nothing free to absorb an error.

Run with `python test/reference/run_cibauto.py`.

## 3b. gal×CIB — PASSES to 0.27%, and the residual is explained

**Closed.** The comparison could not be made against the checked-in
`data/PlanckCIB*GHzxDESI_ELGgalaxy_rl*.txt` (2024-08-01): those outputs predate
every harmonised input by 9–15 months (see the retraction below). The fix was to
re-run the *current* DopplerCIB on the shared grid.

### Regenerating the reference

`test/reference/gen_galxcib_reference.py` runs DopplerCIB in-process.
DopplerCIB itself is **never edited** — `test/reference/doppler_shim.py`
redirects its hardcoded paths (two stale generations of them) and restores
`np.trapz` / `scipy.integrate.simps`, which newer numpy/scipy dropped. Output:
`data/cl_galxcib_dopplercib_planck.npz`.

The shared grid is not a free choice; the shared files pin it:

| file | shape | implies |
|---|---|---|
| `tmp/hmf.npy`, `tmp/bnu.npy` | (100, 20) | 100 masses, 20 redshifts |
| `tmp/pz.npz['z']` | (20,) | `np.linspace(0.5, 1.5, 20)` |

and galCIB reproduces `tmp/hmf.npy` **bit-exactly** with `Mh = np.logspace(7, 15, 100)`
on that redshift grid, which pins the mass grid.

Two variants are written, because DopplerCIB is internally inconsistent about
which HMF the cross uses: `Gal_halo.py:18-19` injects the shared arrays, but
`CIBxgal.__init__` calls `Cib_halo.__init__` *after* `ProfHODMore15.__init__`
and `CIB_halo.py:51-55` overwrites both with its own. It turns out not to
matter — DopplerCIB's own HMF equals `tmp/hmf.npy` to `1.000000`, and its bias
to 0.15–0.35%, moving the 2-halo term by 0.16%.

### Result — galCIB / DopplerCIB

Compatibility mode: `limber_offset=0` (k=ℓ/χ), `hmalpha=1`, no `W_mu`, no colour
correction (DopplerCIB applies none in the cross, `CIBxGal_halo.py:90-115`,
unlike its auto), pure-NFW profile shared between galaxies and CIB, `Ncen_IR=1`.

| ν (GHz) | 1-halo | 2-halo | total |
|---|---|---|---|
| 353 | 0.9977 | 0.9971 | 0.9973 |
| 545 | 0.9977 | 0.9971 | 0.9973 |
| 857 | 0.9976 | 0.9972 | 0.9973 |

**Flat in ℓ** (identical to 4 decimals at ℓ=100, 600 and the 200<ℓ<2000 median)
and **flat in frequency**. Reproduce with `test/reference/run_galxcib.py`.

### The residual 0.27% is a cosmology-library convention, not a bug

Term-by-term, galCIB / DopplerCIB:

| quantity | ratio |
|---|---|
| `nbar` (galaxy HOD ⊗ HMF) | **1.000000** |
| `chi` | 0.99997 |
| M21 SED `snu_eff` | 1.0000 ± 0.0007 |
| `dchi_dz` | 0.9980 |
| `Wg = p(z)/dchi_dz` | 1.0020 |
| `geom = dchi_dz/χ²` | 0.99806 |
| **`djc` (emissivity)** | **0.9967 – 0.9986** |

`dchi_dz` cancels exactly in `geom × Wg = p(z)/χ²`, so the whole residual sits in
the emissivity. Its source: **galCIB takes Ω_m from colossus, DopplerCIB from
astropy**, and the two libraries genuinely disagree on Planck18 —

    colossus  Om0 = 0.31110      (total matter, massive neutrinos included)
    astropy   Om0 = 0.30966      (neutrinos carried as a separate component)
                                  0.465% apart

Ω_m enters the baryon accretion rate twice, with partial cancellation:

    BAR ∝ f_b × √(Om0(1+z)³ + Ode0),   f_b = Ob(z)/Om(z)
    f_b ratio         = 0.99598   (down)
    √(...) ratio      = 1.00186   (up)
    net               = 0.99737 – 0.99800   →  0.20–0.26% deficit

Observed: 0.23% (1-halo), 0.29% (2-halo). **Fully accounted for.** The extra
~0.05% in the 2-halo term is interpolation — DopplerCIB interpolates both the
bias and P(k) from its 210-point redshift grid onto the 20-point one.

This is a deliberate difference, not an error: galCIB is self-consistent in using
colossus for Ω_m, the HMF and the bias.

### Hypotheses tested and falsified

- **`u_cen`**: DopplerCIB gives centrals an NFW profile in the cross
  (`CIBxGal_halo.py:93-94`), galCIB uses `u_cen=1`. Effect measured at **<0.02%**
  — ELG-hosting halos are low-mass, so `u(k)≈1` over this ℓ range and the two
  conventions are indistinguishable. galCIB's choice is intentional.
- **Halo bias implementation** — matches colossus exactly.
- **M21 SED table** — matches exactly.
- **Extra `Ncen_IR`** (galCIB follows paper Eq. 2.40, DopplerCIB has no IR HOD) —
  neutralised exactly by `log10Mmin_IR = 1.0`, verified `Ncen_IR = 1.0000000000`.

### RETRACTED: the old offsets were a configuration mismatch

An earlier version of this log reported 1-halo / 2-halo offsets of ~8% / ~21%
against the 2024-08 reference, attributed them to cached `hmf.npy` / `bnu.npy`
differing from live colossus, and called the diff "blocked" on those files.
**Both claims were wrong**, and the offsets themselves were an artifact.

The files were never missing — they are in `./tmp/` (the earlier search only
looked for DopplerCIB's hardcoded `/Users/tkarim/...` paths). More importantly,
the artifact chronology shows the old reference cannot have used the shared
inputs at all:

| date | artifact |
|---|---|
| **2024-08-01** | old gal×CIB reference (`PlanckCIB*GHzxDESI_ELG*.txt`) |
| 2024-10-15 | `data/gal/dndz_extended.p` |
| 2025-05-01 | shared HMF + P(k) (`*_DopplerCIB.p`) |
| 2025-08-07 | `tmp/hmf.npy`, `tmp/bnu.npy`, `tmp/pz.npz` |
| 2025-11-11 | CIB×CIB reference (`cl_cib_dopplercib_planck.npz`) |
| **2026-08-15** | **new gal×CIB reference** (`cl_galxcib_dopplercib_planck.npz`) |

That diff ran on 91 masses × 210 redshifts with `dndz_extended.p` (30 points,
0.05–2.95); the reference was built on 100 × 20 over z = 0.5–1.5 with a
different n(z) and pre-harmonisation P(k)/HMF. Regenerating on the shared grid
collapses 8%/21% to 0.23%/0.29%. The `√(2h/1h)` consistency across frequency was
a real observation, but the inference that it implied a ~10% bias error was
unsupported — the bias was never wrong.

The old `PlanckCIB*.txt` files are kept for provenance only. **Do not diff
against them.**

---

## 4. Bugs fixed

### 4.1 `compute_Puv_tot` operator precedence — `powerspectra/utils.py`

```python
P_tot = (pk1h**hmalpha + pk2h**hmalpha)**1/hmalpha   # WRONG: parses as (...)/hmalpha
P_tot = (pk1h**hmalpha + pk2h**hmalpha) ** (1.0/hmalpha)   # fixed
```
`**` binds tighter than `/`. Correct at `hmalpha=1` by luck — which is why the
e2e test never caught it — but `analysis/theory.py` passes 0.7, where it gave
5.403 instead of 6.689. Signature also reordered to `(pk1h, pk2h)` to match how
`pk.py` actually calls it.

### 4.2 Y23 SED missing χ²(1+z) — `cib/snumodel.py`

Paper Eq. 2.43: `S_eff[(1+z)ν, z] ∝ Θ[(1+z)ν, z] / (χ²(1+z))`. `snu_Y23`
returned `theta_normed*L0` with no division while `_compute_djc` multiplies by
`χ²(1+z)/K`, so the Y23 emissivity carried a spurious χ²(1+z). The legacy code
divided it out explicitly (`clustering/cib.py::Seff`).

Validated against the M21 path (itself verified exact) — both must share a
geometric convention:

| ν | Y23/M21 flatness, fixed | unfixed |
|---|---|---|
| 353 | 1.7× | 1839× |
| 545 | 1.3× | 1409× |
| 857 | 1.2× | 923× |

χ²(1+z) spans 1076× over 0.1<z<4, matching the unfixed discrepancy. The residual
1.2–1.7× is the genuine grey-body-vs-B15 shape difference, small against the
SED's own 63–354× dynamic range.

### 4.3 Y23 bandpass integrated in the wrong frame — `cib/snumodel.py`

`apply_filter_to_sed` was given `nu_prime` (rest frame) as the integration
abscissa while interpolating at observed Planck filter frequencies, i.e.
evaluating the SED at ν_filt/(1+z). `SnuModel` now retains `nu_obs_grid` and
integrates in the observed frame.

Isolated effect, **factor by which the SED changes**:

| ν | z=0.01 | z=0.43 | z=1.02 | z=3.03 |
|---|---|---|---|---|
| 353 | 1.04 | 3.41 | **10.6** | 490 |
| 545 | 1.04 | 3.01 | **7.92** | 35.1 |
| 857 | 1.03 | 2.42 | **4.84** | 9.11 |

Ratio → 1 as z→0, as it must. **At the DESI ELG redshifts (0.8<z<1.6) this was a
factor 5–10 error.**

### 4.4 Smaller fixes

- `_compute_subhalo_mf` raised `UnboundLocalError` when a custom SHMF was passed
  (`m_over_M` bound inside the `if`) — `cib/cibmodel.py`.
- `get_k_range` / `get_Mh_range` inverted the h convention relative to
  `_load_colossus` — `cosmology/cosmology.py`.
- `cib/default_sfr.py` cited Eq. 2.39 of the paper for what is actually the
  M21/DopplerCIB form. The paper uses `σ₀(1 − τ/z_c·max(0,z_c−z))`; both codes
  use `σ₀ − τ·max(0,z_c−z)`. Kept the DopplerCIB form, corrected the comment.
- **`Nsat_ELG` inf hazard fixed** (`galaxy/default_models.py`). `np.where`
  evaluates both branches, so `((M−M0)/M1)**α` was computed for `M ≤ M0` too;
  with `α_sat = −0.198` that is `0**negative → inf`, and a negative base with
  non-integer α gives nan. The mask discarded them but it raised RuntimeWarnings
  and would surface under a sampler exploring `M0`. Now substitutes a safe
  placeholder base before exponentiating. Verified under warnings-as-errors for
  both fiducials (α = +0.59 and −0.198) and at the exact `M == M0` boundary.
- `test/end2end` used `npt.assert_array_equal` (bit-exact) against stored
  regression files. It failed at 4.1e-4 purely from camb/numpy version drift —
  **confirmed pre-existing by stashing all edits and re-running**. Converted to
  `assert_allclose(rtol=1e-3)`.

---

## 5. Consequence of the Y23 fixes — read before using any Y23 result

**Any Y23 output produced before these fixes carries both errors**, including the
work under commit `4206d50 updated Y23 model`. The M21 path is unaffected.

The pre-fix regression files are preserved for provenance as
`test/end2end/cgI_prefix.txt` and `cII_prefix.txt` ("pre-fix", not a string
prefix). Current references regenerated with the fixed code; the suite passes.

| | new / pre-fix | note |
|---|---|---|
| `cgI` | 3.15e-07 | one power of the SED |
| `cII` | 5.92e-14 | two powers — and (3.15e-7)² = 9.9e-14, consistent |

`cgg` is unchanged (no SED dependence).

**`L0` must be re-tuned.** It is a pure normalisation, so no science conclusion
depends on its value, but the fiducial and prior range are now wrong by ~3.2e6:
old `L0 = 5e-14` corresponds to `L0 ≈ 1.6e-07` under the fixed code.

---

## 6. Parameter-set findings

### γ_dust is *exactly* degenerate with L0 — fix it

Two checks, which initially appear to disagree:

- **Analytically γ is unreachable.** The `ν^−γ` branch applies only where
  ν_rest ≥ ν₀(z). Over 0.8<z<1.6 the channels sit at 635–2228 GHz rest-frame
  while ν₀(z) = 3323–3590 GHz, so all three Planck bands are always in the
  modified-blackbody branch and that branch is never evaluated.
- **Yet snu_eff moves 10–130% with γ**, because γ enters ν₀(z) via the Lambert-W
  solution and ν₀ is the *normalisation point* (`theta/theta(nu0)`).

Resolution — the effect is a pure rescaling:

| parameter | median effect | spread over (ν,z) |
|---|---|---|
| γ_dust 1.7→1.2 / 2.5 / 4.0 | 0.892 / 1.269 / 2.292 | **1.000000** |
| β_dust 1.5 / 2.5 (contrast) | 1.422 / 0.651 | 1.716 / 1.796 |

A spread of exactly 1.000000 means γ changes only the amplitude, so it is
**exactly degenerate with L0**. Sampling both opens a perfectly degenerate
direction that will wreck the sampler's conditioning. Fix γ = 1.7 — not because
the paper does, but because it carries no independent information here.
β_dust is a genuine shape parameter and should stay free.

## 7. Open items

1. **Fisher forecast** against the measured covariances
   (`data/measurements/gal_cib/snr_*.npz`) before any MCMC.
2. `analysis/theory.py` is still stale against the current API. (`Sampler` was
   rewritten — see §4.4 — and is exercised by `test/test_likelihood.py`.)
3. Decide whether `include_cgg` should default on, and whether to wire the real
   CII measurement (`data/measurements/cl_cibxcib_fullmission_beam_corrected.npy`)
   into `run_end2end.py`.

---

## 8. Reproducing

```bash
python test/reference/gen_galxcib_reference.py   # re-run DopplerCIB -> reference
python test/reference/run_galxcib.py             # gal x CIB diff, 1h/2h split
python test/reference/run_cibauto.py             # CIB x CIB diff, 21 pairs
python test/reference/run_end2end.py             # measured data -> model -> likelihood
python -m pytest test/ -q                        # regression suite
```

`test/reference/harness.py` holds the shared config: the DopplerCIB parameters
(`CIB_halo.py:41-47`, `Gal_halo.py:29-69`), the reference loaders, and the P(k)
injection. `test/reference/doppler_shim.py` is what lets DopplerCIB run
unmodified — path redirection plus `np.trapz` / `intg.simps` aliases.

`test/reference/run_tier1.py` is superseded by `run_galxcib.py`: it diffs
against the retracted 2024-08 reference and is kept only to reproduce the
retraction in §3b.
