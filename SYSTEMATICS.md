# DESI DR2 ELG × Lenz19 CIB — data systematics status

Handoff notes for continuing the systematics work on the angular cross-power
spectrum measurement. Everything below is measured from the data in this repo
unless marked otherwise. Numbers are quoted so they can be re-checked.

**Status: the measurement pipeline is validated and working. What remains open
is mostly low-ℓ imaging systematics and a few quantifications listed in §6.**

---

## 1. Setup

| | |
|---|---|
| Galaxies | `data/dr2/ELG_LOPnotqso_clustering.dat.fits`, 7,482,654 rows |
| Randoms | `data/dr2/ELG_LOPnotqso_{0,1}_clustering.ran.fits`, 51,725,602 rows total |
| CIB | `data/cib/cib-lenz19-data/{353,545,857}/4.0e+20_gp40/`, nside 2048, RING, **Galactic**, MJy/sr |
| Code | `analysis/measure_gal_cib.py`, `analysis/validate_gal_cib.py` |
| Env | `/opt/anaconda3/envs/cib`, **NaMaster 3.0**, healpy 1.19 |
| Outputs | `data/measurements/gal_cib/` |

Catalogs are rotated ICRS→Galactic once and cached to
`data/measurements/gal_cib/cache/cat_{data,rand}.npz`. Σw_data = 1.355e7,
Σw_rand = 1.292e8, α = 0.105.

Tomographic bins (dz=0.2): 1,788,304 / 1,846,180 / 1,675,902 / 1,229,302 galaxies.

```bash
python analysis/measure_gal_cib.py --stage cats            # build catalog cache
python analysis/measure_gal_cib.py --routes A B C          # measure (182 s)
python analysis/validate_gal_cib.py --snr --snr-plot --fsky --shotnoise \
                                    --spectra --jackknife --null
# config flags on both: --dz {0.2,0.4,0.8} --lmax --lmax-mask --nran
```

### Three estimator routes

Deliberately a ladder, so a discrepancy localises to one step:

| route | data positions | footprint | class |
|---|---|---|---|
| **A** (fiducial) | exact catalog | exact random catalog | `NmtFieldCatalogClustering(pos_d, w_d, pos_r, w_r)` |
| **B** | exact catalog | pixelised random density | `NmtFieldCatalogClustering(..., mask=)` |
| **C** | pixelised counts | pixelised random density | `NmtField(mask, [delta_g])` |

A→B isolates pixelising the footprint; B→C isolates pixelising the data.

---

## 2. Settled: estimator validation

- **Routes agree after decoupling.** Over 100<ℓ<600, all 12 (bin,ν) pairs:
  B/A = 0.8–1.9%, C/A = 1.3–3.7%. First bandpower B/A = 0.998–1.003.
- **`lmax_mask` converged.** 3000 / 4000 / 5000 agree to 0.0000. Fiducial 4000.
- **NaMaster 2.7 → 3.0 regression:** max |ratio−1| = 8.9e-16 over all 36 spectra.
- **463 of 468 bandpowers positive** (all 5 negatives at 353 GHz, ℓ≥826). A sign
  test with no covariance assumption — strong evidence the high-ℓ signal is real.
- **Null test** (rotate data *and* randoms by 120° in ℓ): χ²/dof = 0.86–1.15 above
  ℓ=600, 0.48–0.93 below; median |null|/|signal| = 0.127.
- **Jackknife vs Gaussian covariance** (56 patches, nside_jk=4, bin 0.8–1.0 × 545):
  median σ_JK/σ_G = **1.04** (100<ℓ<600), **1.10** (600–1200), **1.12** (1200–2000).
  The monotonic rise is the expected trispectrum signature. Inflate Gaussian
  errors ~10% above ℓ=600. Diagonal only — 56 patches vs 39 bandpowers gives a
  Hartlap factor of 0.27, so the full JK matrix is not invertible.

---

## 3. Settled: scales

### fsky and ℓ_min

fsky_g = 0.3055–0.3075, fsky_I = 0.299–0.319, **fsky_eff = 0.126–0.142**
(`⟨w_g w_I⟩²/⟨w_g² w_I²⟩`, ~5200–5900 deg²). Note ⟨w_g·w_I⟩ = 0.031–0.043 versus
fsky_g×fsky_I ≈ 0.096 — the DESI and Lenz footprints overlap much less than
either alone, and that overlap is the dominant limit on total S/N.

**Mode-coupling is not the ℓ_min limitation.** Adjacent-bandpower correlations
are −0.027 at the first bandpower, falling to −0.007 by ℓ≈130; ~380 effective
modes already in bp0 (ℓ∈[2,51]). ℓ_min should be set by *imaging systematics*
(see §6), provisionally ℓ_min ≈ 50–100.

### ℓ_max — beam-limited, not statistics-limited

`Wl_eff = pixfunc × Bl_eff` (verified: ℓ=1000, 0.98893×0.82585=0.81671). Pass it
as `beam=` and it deconvolves the 5′ beam *and* the HEALPix pixel window at once —
do **not** additionally divide by `hp.pixwin`.

**`Bl_eff` hits a hard floor of 0.00255 at ℓ=3988** (identical at all three
frequencies — a regularisation floor, not physics), so 1/Wl_eff jumps 9.1 → 405.
**lmax = 2×nside = 4096 is unusable.** Practical ceiling **lmax = 3000**
(deconvolution 5.9×).

Going 2000→3000 gains +6% (353), +14% (545), **+18% (857)** in total S/N.

### Shot noise

`N_shot` matches analytic `1/n̄` to 3–4 digits in all 7 configurations.
**The galaxy auto crosses below shot noise at ℓ ≈ 76** in every configuration.

| dz | bin | N_shot |
|---|---|---|
| 0.2 | 0.8–1.0 … 1.4–1.6 | 8.06e-6, 7.65e-6, 8.50e-6, 1.107e-5 |
| 0.4 | 0.8–1.2, 1.2–1.6 | 3.93e-6, 4.81e-6 |
| 0.8 | 0.8–1.6 | 2.17e-6 |

**This does not bias the cross-spectrum** — galaxy Poisson noise is uncorrelated
with the CIB. It enters only the variance, as √N_shot. Above ℓ≈76 where
C^gg,tot ≈ N_shot is flat, mode count (2ℓ+1)Δℓ grows linearly and nearly cancels
the signal decline, which is why per-bandpower S/N stays ~2–4 across the range.

Confirmed independently: widening bins cuts shot noise 4× and leaves the fraction
of S/N above ℓ=600 unchanged at 0.4–0.6.

### Recommended cuts

**ℓ ∈ [100, 1000] for 353 GHz; [100, 2000] for 545/857**, computed at lmax=3000,
Gaussian errors inflated ~10% above ℓ=600. Cutting at 400–600 would discard
21–40% of total S/N.

---

## 4. Settled: fiber assignment

Two **distinct** systematics; they are handled by different mechanisms.

**(a) Priority ordering (QSO > LRG > ELG).** A large-scale angular modulation of
the ELG selection. Handled because `WEIGHT_COMP` is *true IIP*:

| candidate | median ratio | frac within 1% |
|---|---|---|
| `WEIGHT_COMP` vs 1/`PROB_OBS` | **0.99839** | **85.2%** |
| `WEIGHT_COMP` vs 1/`FRAC_TLOBS_TILES` | 1.130 | 12.8% |

`PROB_OBS` comes from alt-MTL realizations — repeated runs of the actual DESI
assignment code, which contains the real priority ladder and real collisions. So
both effects are captured at object level. This is better than DR1, which used
1/f_TLID. NTILE is 3–5 over 75% of the footprint (only 4.2% single-pass).

**(b) Patrol-radius collisions.** DESI: *"fiber assignment incompleteness affects
clustering below θ ∼ 0.05°"* → **ℓ ≈ 3600**, above our whole usable range.

**No θ-cut analogue is needed.** The 3D P_ℓ(k) θ-cut exists because Fourier modes
mix all angular separations; an angular cross-spectrum has no galaxy-galaxy pairs.
Per 2505.20656 §3.2: completeness weights *"are adequate for correcting for
fiber-assignment"* for C_ℓ^κg, and PIP *"doesn't really have any meaning for e.g.
C_ℓ^κg"* — **do not run PIP/bitweights here**, they correct pairs, not the density.

**Residual risk, measured.** The completeness field *does* correlate with the CIB:

| bin | A(comp×CIB)/A(gal×CIB) | r(comp, CIB) |
|---|---|---|
| 0.8–1.0 | 0.107 | −0.160 |
| 1.4–1.6 | 0.041 | −0.149 |

Negative sign is the expected mechanism (crowded regions lose ELGs to
higher-priority targets; crowded regions are CIB-bright). A fractional error ε in
the weights biases the cross-spectrum by roughly ε×0.1, so ~20% weight accuracy
keeps it under 2%. **Optional hardening:** pass the completeness map as
`templates=` to `NmtFieldCatalogClustering` (with `lmax_deproj`) and let NaMaster
deproject it — cheaper and more targeted than a θ-cut.

### FKP: recommend NOT applying

`WEIGHT` does **not** include FKP (no `WEIGHT_FKP` column); DESI says multiply
`w_tot × w_FKP` if wanted. P₀,ELG = **4000** (Mpc/h)³.

| bin | n̄P₀ | radial w_FKP spread |
|---|---|---|
| 0.8–1.0 / 1.0–1.2 / 1.2–1.4 / 1.4–1.6 | 1.15 / 1.00 / 0.80 / 0.55 | 9.3% / 11.5% / 7.7% / 20.8% |
| 0.8–1.6 | 0.87 | 42.7% |

Three reasons to skip at dz=0.2: (i) a constant weight cancels in δ_g, and the
within-bin variation is only 8–21%, so the variance gain is second-order;
(ii) `w_FKP(z, n_tile)` depends on NTILE, which varies across the sky — it would
imprint an *angular* modulation on an angular measurement; (iii) **FKP balances
sample variance against shot noise, but we are shot-noise dominated above ℓ=76**,
where the optimal weight is uniform. DESI explicitly invites analyses at other
scales to derive their own weighting.

---

## 5. Traps — mistakes already made, do not repeat

1. **Coupled vs decoupled pseudo-Cl.** The original "catalog ≠ map" puzzle. A
   catalog field's mask is `Σ_j w_j δ(n−n_j)`, **unnormalised by solid angle**, so
   its pseudo-Cl sits ~1e16 above a map-based one. `NmtWorkspace` absorbs this
   exactly. **Only ever compare decoupled C_ℓ.**
2. **CIB mask baked into the map.** `NmtField(np.ones_like(map_eff), [map_eff])`
   leaves the CIB footprint never deconvolved. `masked_on_input` defaults to
   False, so pass the *raw* NaN-cleaned map plus a separate mask.
3. **Pooling randoms across z for the mask is wrong.** fsky matches (0.3155 vs
   0.3165 at nside 256) but the completeness *pattern* does not: normalised all-z
   vs per-bin random densities correlate at only **0.68** at nside 512, 55% of
   pixels differing >20%. Pooling shifts the first bandpower by −3.7×. **Use
   per-bin randoms.**
4. **Randoms are too sparse to bin at nside 2048.** Per-bin, that gives 1.51
   randoms/occupied pixel, 62% singletons, and recovers only fsky 0.163 of the
   true ~0.31. Hence `MASK_NSIDE=512` (12.8/pixel). Convergence of B onto A:
   nside 256 → 6.1%, 512 → 1.15%, 1024 → 0.75%.
5. **Null test must rotate data *and* randoms.** Rotating only the data leaves
   galaxies outside their own footprint — a spurious −275 first bandpower.
6. **P₀,ELG = 4000, not 10⁴** (10⁴ is the LRG value).
7. **Random `WEIGHT` is a global constant × (COMP×SYS×ZFAIL)** — ratio 1.4561,
   identical across all NTILE 1–7, so it cancels in α. An earlier claim that
   "random weights don't decompose" was wrong.
8. **Only 2 random catalogs are on disk**, not 17. Sensitivity: 1→2 files changes
   route-A C_ℓ by 5.0% (ℓ<600) / 18.7% (ℓ>600), implying ~5% random-induced error
   now and ~1.7% with 17 files — a **~1% improvement in the error budget**. Not
   worth it for route A; would matter for B/C, where `MASK_NSIDE` is capped by
   random sparsity.

---

## 6. Open items

1. **Low-ℓ imaging systematics — the main gap.** `WEIGHT_SYS` runs 0.69–2.0
   (std 0.11). Not tested. The null test rotates only in Galactic longitude, so it
   preserves |b| and is **blind to dust-gradient systematics** — the most likely
   low-ℓ contaminant, given the Lenz maps are N_HI-selected. Needed: a null that
   also rotates in b, and/or a `WEIGHT_SYS`-off comparison. Until then ℓ_min ≈
   50–100 is a judgement call, not a result.
2. **Fiber residual never quantified on this data.** The literature argument (§4)
   is solid but it is an argument. Two attempts to measure it were invalid and
   retracted. Given trap #7, a corrected test is now feasible.
3. **Non-Gaussian covariance.** Jackknife gives 1.04–1.12 for one (bin,ν) pair
   only. Worth repeating at 857 GHz and in the highest-z bin, where the 1-halo
   term is strongest.
4. **Tomography adds little raw S/N** — quadrature over dz=0.2 bins vs one wide
   bin: 36.9 vs 34.5 (353), 47.1 vs 44.8 (545), 52.4 vs 50.8 (857), i.e. 3–7%. And
   that *overstates* it, since narrow bins share the same CIB map and are
   correlated. Value of tomography is redshift evolution, not significance.
5. **Notebook not migrated.** `notebooks/data-measure.ipynb` cells 39–100 still
   contain the original buggy code (CIB mask bug, binary galaxy mask, z-key
   rounding bug, no workspace caching). The scripts supersede it.
6. **857 GHz is beam-limited**: ℓ(90%) at lmax=3000 lands at 1926–2326, i.e. still
   accumulating where the beam cuts off.

---

## 7. Key references

- [DESI 2024 II](https://arxiv.org/pdf/2411.12020) — §5.3 w_comp, §8.2 Eq. 8.2–8.4
  w_tot and w_FKP definitions, P₀ values. **The weight-definition source.**
- [arXiv:2505.20656](https://arxiv.org/pdf/2505.20656) §3.2 — DESI DR1 galaxy ×
  CMB lensing; states completeness weights suffice for cross-correlations and PIP
  is meaningless for them. Closest methodological analogue.
- [arXiv:2406.04804](https://arxiv.org/pdf/2406.04804) — θ-cut estimator;
  θ ∼ 0.05° scale, DR1 ELG completeness ~35%.
- [arXiv:2411.12025](https://arxiv.org/pdf/2411.12025) — fiber assignment
  incompleteness characterisation.
- [CIB tomography with SDSS/BOSS/eBOSS](https://iopscience.iop.org/article/10.3847/1538-4357/adfb6a)
  — closest spec-z × CIB precedent; real-space w(θ), does not treat fiber
  collisions. DESI × CIB appears not yet done.
