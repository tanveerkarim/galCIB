# Fiber assignment in DESI DR2 ELG × CIB — are the DESI weights enough?

Companion to `SYSTEMATICS.md`. Scope: fiber collisions and fiber assignment
incompleteness only. Imaging weights are touched on once, in §9, because they
interact with the CIB in a way that is specific to this measurement.

**Bottom line.** `SYSTEMATICS.md` §4 currently records this as *settled*. It
should be downgraded to *argued but unmeasured*. The argument is sound in
mechanism and the DR2 weights are materially better than the DR1 ones that the
literature claim was made for — but the ALTMTL-vs-complete mock figures show a
5–25% scale-dependent deficit in `Cl_kg` whose provenance is currently unknown,
and until that is resolved the fiber budget is unbounded. §7 gives a ladder of
tests, the first four of which are cheap and run on data already on disk.

---

## 1. The question, stated precisely

The estimator is linear in the galaxy field:

```
Cl^{g,CIB}  =  < delta_g(l,m)  I*_CIB(l,m) >
```

so the entire fiber burden falls on one requirement: **the weighted galaxy
counts must be an unbiased estimate of the target density field, at every
angular scale used.** Nothing else about fiber assignment can enter. There are
no galaxy–galaxy pairs in this estimator, so no pair-counting correction —
PIP, θ-cut, nearest-neighbour upweighting — has any direct analogue here.

That is a real simplification and `SYSTEMATICS.md` is right about it. But it
cuts both ways: because there is no pair-level fallback, the weights are
*load-bearing* in a way they are not for `P_l(k)`. In the 3D analysis, a
θ-cut removes the regime where weights are known to fail. Here there is no
such escape hatch, only the weights.

---

## 2. What `w_comp` actually is in DR2, and why it beats DR1

DESI defines two completeness components ([2411.12020] §5.1–5.3):

| quantity | meaning | captures |
|---|---|---|
| `f_TLID` | 1 / (number of unique targets competing for that tile+fiber) | close-pair / patrol-radius competition |
| `f_tile` | completeness per tile-group `t_group` (analogous to SDSS sector `C_BOSS`) | tile-level budget, priority veto |
| `p_obs = N_assign/129` | assignment probability from 128 altMTL reruns + the real one (Eq. 5.1) | **everything the real `fiberassign` does**, per object |

DR1's *fiducial* choice was the crude one, `w_comp = 1/f_TLID` (Eq. 5.2), with
`f_tile` pushed onto the randoms. DESI itself recommends the altMTL version for
small-scale clustering.

Your DR2 catalog carries the good one. `SYSTEMATICS.md` measures
`WEIGHT_COMP` vs `1/PROB_OBS` at median ratio 0.99839, 85.2% within 1%,
against 1.130 / 12.8% for `1/FRAC_TLOBS_TILES`. So `WEIGHT_COMP` is true IIP.

**Why that matters mathematically.** If selection is Bernoulli with probability
`p_i`, conditionally independent of the density field *given* `p_i`, then

```
E[ sum_i s_i / p_i ]  =  sum_targets p_i * (1/p_i)  =  N_targets      (exact)
```

— unbiased at *every* scale, with no scale-dependent residual at all. And the
conditional-independence assumption is not an approximation here: altMTL reruns
`fiberassign` on the *actual* target catalog, so `p_i` is already conditioned on
the real local crowding, the real QSO>LRG>ELG priority ladder, and the real
hardware state. This is the strongest form of the "weights are enough"
argument, and it is available to you but was *not* available to DESI DR1 or to
[2505.20656].

**Where the identity breaks.** Four places, and they are the whole risk:

1. **`p_i = 0` objects.** Not up-weightable — they are absent from the
   catalog and no weight can resurrect them. [2411.12025] reports ~2.5×10⁵
   zero-probability ELG targets in the NGC alone, and stresses that they are
   *clustered*. In a hard pair where only one member can ever be reached, the
   pair contributes one galaxy, the survivor gets weight ≈1, and the local
   density is permanently halved. This depletes exactly the high-density peaks.
2. **`p̂` estimator bias.** DESI does *not* use the naive `1/p̂ = 129/N_assign`.
   The DR2 ELG papers define the IIP weight as a shrinkage estimator,
   `w_comp,IIP = (1+128)/(1+popcount(b))` = `129/(1+N_assign)`, which bounds the
   weight at 64.5 and kills the Jensen blow-up as `p → 0`. The price is a small
   *downward* bias: `E[129/(1+N)] ≈ (1/p)·129/130 ≈ 0.99/p` for moderate `p`,
   growing as `p` falls. So the residual is a mild **under-correction
   concentrated in low-completeness, high-crowding regions** — right sign and
   right spatial pattern to contribute to the mock deficit, but ~1%, far too
   small to explain 25% on its own. This is better-behaved than I first assumed
   and is not the main risk; items 1, 3 and 4 are.
3. **Data/random weight consistency.** `WEIGHT` is `w'_tot / <w_comp>(n_tile)`
   ([2411.12020] Eq. 8.2) — the NTILE-dependent mean completeness is divided out
   of the data. It must be divided out of the randoms identically or an
   NTILE-shaped angular modulation is injected. Trap #7 in `SYSTEMATICS.md`
   records the random `WEIGHT` ratio as a *global* 1.4561, "identical across all
   NTILE 1–7". If that really is NTILE-independent while the data side is not,
   there is a mismatch. **This needs checking before anything else** (test T1).
4. **The 15% of objects where `WEIGHT_COMP` ≠ `1/PROB_OBS` by >1%.** Almost
   certainly the low-`p` tail, i.e. the same objects as (1) and (2).

Note what these four have in common: **they all concentrate in low-completeness,
high-target-density regions.** Those regions track the tiling pattern (NTILE)
and the Galactic structure that drives target density. So the residual, if it
exists, lives at **low ℓ and in the additive term** — not at ℓ≈3600 where
`SYSTEMATICS.md` §4 currently places it. The "ℓ≈3600, above our whole usable
range" argument correctly disposes of the *patrol-radius* effect and is not
wrong, but it addresses the wrong failure mode.

---

## 3. The θ-cut: correctly judged not applicable, for a reason worth writing down

[2406.04804] exists because `P_l(k)` and `ξ_l(s)` are *quadratic* in the field:
a Fourier mode at k=0.2 h/Mpc receives contributions from pairs at all angular
separations, including θ < 0.05° where DESI physically cannot observe both
members. The fix is to truncate those pairs from the estimator and put the same
truncation into the window matrix.

`Cl^{g,CIB}` is linear in `delta_g`. There are no galaxy pairs. There is nothing
to truncate. **The θ-cut has no analogue and should not be constructed.**
`SYSTEMATICS.md` §4 is correct here and this should stay in the paper.

Two secondary points worth keeping straight:

- The angular scale of the patrol-radius effect is θ ≈ 0.05° → ℓ ≈ 180°/θ ≈
  **3600**, well above your ℓ_max of 1000–2000. Even if a `Cl_gg`-style
  suppression existed it would not reach you.
- [2411.12025] finds that weighting schemes agree only above ~20 h⁻¹Mpc, with
  residual suppression up to 60% for ELGs below θ ≈ 0.02°. Twenty h⁻¹Mpc at
  z=1.2 (χ ≈ 2620 h⁻¹Mpc) is θ ≈ 0.44°, i.e. **ℓ ≈ 400–800** depending on
  convention — which *is* inside your range. But that statement is about
  pair statistics, where a separation-s pair can have arbitrarily small r_⊥.
  It does not transfer to a linear angular estimator. Do not quote it as if it
  bounds your measurement; do mention why it doesn't.

---

## 4. What [2505.20656] §3.2 does and does not establish

The relevant sentences:

> "For cross-correlations with lensing (`C_ℓ^κg`) these completeness weights are
> adequate for correcting for fiber-assigment, but not for 3D galaxy-galaxy
> spectra."

> PIP "does not actually modify the galaxy density itself but just corrects
> pairs entering the 2-point function estimator, and thus doesn't really have
> any meaning for e.g. `C_ℓ^κg`."

> (fn. 9) "both methods involve weighting each galaxy to give a modified density
> field for which the large-scale impact of fiber assignment on `δ_g` is
> removed."

**In your favour:** the mechanism is right, the PIP point is right and worth
citing, and it is the closest published methodological analogue (same
`NmtFieldCatalogClustering` / DirectSHT estimator, same DESI spectroscopic
samples, same class of external map).

**Against relying on it:**

1. **It is an assertion, not a measurement.** There is no mock validation of
   `C_ℓ^κg` under fiber assignment anywhere in that paper. Their Table 2 lists
   "Comp. weights and θ-cut" against Fiber Assignment with the caption noting
   the first four rows are for the *3D* power spectrum. The angular rows carry
   pixelization and κ-normalization only. The `C_ℓ^κg` fiber claim is
   unaccompanied by a number.
2. **Different weights.** DR1, `1/f_TLID` — the crude proxy, not IIP. The
   footnote acknowledges IIP is "a similar but more robust version" that they
   could not afford. Your DR2 catalog has the robust one, so the *claim* is
   weaker than what you can actually justify — but their *evidence* for it is
   correspondingly weaker too.
3. **Different precision regime.** Their `C_ℓ^κg` runs to ℓ ≈ 300–600 with
   error bars visibly ~20–50% per bandpower (their Fig. 3, top row), and they
   note the κ-normalization correction alone is ≲50% of the `C_ℓ^κg` errors.
   A 10% fiber residual is invisible at that precision. You have total S/N
   36.9–52.4 and per-bandpower S/N 2–4 out to ℓ=2000, so 5% is ~2σ on the
   amplitude and a 20% tilt across your range is fatal. **"Adequate" is not a
   scale-free statement — it is adequate relative to their error bars.**
4. **κ vs CIB.** CMB lensing is essentially uncorrelated with the DESI observing
   footprint. The CIB is not — you have measured `r(comp, CIB) = −0.15` and an
   additive term at 4.1–10.7% of signal. Their conclusion does not transfer to
   the additive channel. See §6.

**Prior spec-z × CIB work does not help.** The closest precedent is the
SDSS/BOSS/eBOSS × CIB tomography ([2025, ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/adfb6a)),
which works in real-space `w(θ)` and does not treat fiber collisions at all.
BOSS was ~95% complete after close-pair weighting; DESI DR1/DR2 ELGs are ~35%
complete. **There is no published spec-z × CIB analysis that has had to solve
this.** You are first, which means you will be asked, and "a lensing paper said
it was fine" is not going to survive referee.

---

## 5. Reading the ALTMTL-vs-complete mock figures

This is the crux and the highest-priority action item.

**What the figures show.** `Cl_kg` (both ACT DR6 and Planck PR4), ALTMTL /
complete: ≈0.92–0.95 at ℓ≈20, ≈1.0 at ℓ≈35, then a smooth monotonic decline to
≈0.85 at ℓ≈200, ≈0.80 at ℓ≈500, ≈0.75 at ℓ≈1000. The GLAM band is tight; the
single SHAM realization scatters around it. This is a **coherent, scale-
dependent, multiplicative suppression across exactly your ℓ range.**

**What is an artifact and should be set aside.** In `Cl_gg`, the ALTMTL curve
crosses zero near ℓ≈700 and the ratio goes negative. ALTMTL has ~35% of the
galaxies, hence ~3× the shot noise; the zero crossing is over-subtraction of an
imperfectly-estimated shot-noise term, not fiber physics. Do not spend time on
the high-ℓ `Cl_gg` panel. `Cl_kg` carries no shot noise and is the only panel
that is clean — which, conveniently, is also the one that matters for you.

**What the mock provenance notes settle.** The suite is **DESI Y3 (= DR2)**,
i.e. the same survey state as your data, and the complete mock has the *same
footprint and selection* as ALTMTL with only fiber assignment switched off.
Two consequences:

- `T(ℓ)` measured on these mocks transfers to your measurement directly — no
  DR1→DR2 extrapolation, no coverage mismatch. This is the right suite.
- Footprint/selection mismatch is **eliminated** as an explanation for the
  deficit. That was one of the two leading artifacts. It does not eliminate the
  other: whether the *randoms* used on the ALTMTL side encode the
  fiber-assignment completeness modulation or only the target selection.
  "Same selection" is a statement about the catalogs, not about the randoms.

Note also that Y3 coverage is far better than the DR1 numbers quoted throughout
this document: `SYSTEMATICS.md` measures NTILE 3–5 over 75% of the footprint
with only 4.2% single-pass, against DR1's 35.2% ELG completeness and only 33%
of area at NTILE ≥ 3 ([2411.12020] Table 6). Zero-probability targets, which
concentrate in single-pass regions, should be correspondingly rarer. Every
DR1-derived risk number in §2 therefore **overstates** your case — which cuts
the wrong way for the mocks: a deficit this large in a *Y3* suite is harder to
explain away, not easier.

**Step zero: were completeness weights applied to the ALTMTL catalogs?** You
were not sure, and the provenance notes do not say. But they record **three**
versions compared while listing only two — ALTMTL and complete. **The third
version is the missing piece, and it is almost certainly the weighted one**
(the standard DESI triplet is complete / ALTMTL raw / ALTMTL + completeness
weights, sometimes with PIP + angular upweighting as a fourth). Recovering that
third label is now the cheapest possible way to answer the whole question — it
may already have been measured.

Everything below branches on it:

- **No weights applied.** Then the figure is measuring *raw* fiber-assignment
  incompleteness and says nothing yet about whether the weights work. Entirely
  consistent with [2411.12025] and with everything in §2. The figure would
  simply be showing you the size of the correction the weights are there to
  make. The next run is weighted-ALTMTL vs complete, and it is the only run
  that answers the question.
- **Weights applied.** Then this is a 5–25% *residual*, it directly falsifies
  the [2505.20656] claim in your ℓ range, and it is the single most important
  number in your systematics budget. It would also be publishable in its own
  right.

Also check, in the same pass: were **per-realization randoms** used for the
ALTMTL measurement, or the complete-mock randoms? Using complete randoms
against a fiber-assigned catalog puts the completeness modulation into `δ_g`
instead of the mask and generates a spurious signal by itself.

**If it does turn out to be a genuine residual, here is the mechanism I would
bet on.** Zero-probability pairs (§2, item 1). In a close pair where only one
member is ever reachable, the survivor's weight cannot recover the lost galaxy.
This preferentially depletes high-density peaks, which suppresses the 1-halo
contribution to `Cl_kg` — and the 1-halo term is exactly what turns on above
ℓ ≈ a few hundred and grows toward small scales. That predicts a smooth
monotonic decline of the observed shape, and it predicts three testable things:
the residual should be **strongest in NTILE=1 regions**, should **scale with
the local zero-probability fraction**, and should be **absent at ℓ ≲ 50**.
All three are checkable (T2, T5).

---

## 6. Can κ mocks stand in for CIB mocks?

You raised the right objection: the Uchuu mocks have galaxies correlated with
CMB lensing, not with the CIB. **Yes, they can — for the part that matters
most.** Write the observed cross-spectrum against any external map `X` as

```
Cl^{X,obs}(l)  =  T(l) * Cl^{X,g,true}(l)   +   Cl^{X,sys}(l)
                  \_____ multiplicative _____/     \__ additive __/
```

**`T(ℓ)` — the multiplicative transfer function.** This is the fiber effect
acting on the *galaxy* field: depletion of peaks, density-dependent selection,
weight-estimator bias. It is a property of the ELG selection function alone. To
the extent that `X` is uncorrelated with the DESI observing strategy — true for
κ, true for the *unmodulated* part of the CIB — `T(ℓ)` is **independent of
`X`**. A κ mock measures it and it transfers to the CIB directly.

Caveats, both manageable:
- `T(ℓ)` is formally z-dependent (completeness varies across n(z)) and κ and
  CIB weight redshift differently. **Measure `T(ℓ)` per tomographic bin**, not
  on the wide 0.8–1.6 sample. At dz=0.2 the residual kernel difference is
  second-order.
- The mock κ is a reconstruction with its own mask and noise. That cancels in
  the ALTMTL/complete ratio *provided the identical κ map, mask and mode-
  coupling matrix are used on both sides*. Worth confirming explicitly.

**`Cl^{X,sys}` — the additive term.** This is `<X · (c − <c>)>/<c>`, the
correlation between the residual completeness field and `X` itself. It is
**strongly `X`-dependent and κ mocks cannot give it to you.** For κ it is
negligible: CMB lensing is near-full-sky and knows nothing about DESI tiling.
For the CIB it is not — you have already measured
`A(comp×CIB)/A(gal×CIB)` = 0.107 (bin 0.8–1.0) and 0.041 (bin 1.4–1.6), with
`r(comp, CIB) = −0.15`. That is a 4–11% additive contamination sitting in your
data right now, with the physically expected sign (crowded fields lose ELGs to
higher-priority targets; crowded fields are CIB-bright).

**So the division of labour is clean:**

| term | measured from | status |
|---|---|---|
| `T(ℓ)` | ALTMTL vs complete κ mocks, per z-bin | not yet done properly |
| `Cl^{X,sys}` | real data + real CIB map (deprojection / direct template) | partially done, needs to be turned into a correction or a bound |

This also means: **do not ask for CIB-correlated mocks.** They would be
expensive, they do not exist, and they are not needed. Ask for exactly one
thing — weighted ALTMTL vs complete `Cl_kg` per tomographic bin — and get the
additive term from your own data.

---

## 7. The test ladder

Ordered by cost. T1–T4 use data already in `data/dr2/` and need no mocks.

`analysis/measure_gal_cib.py::build_catalog_cache` currently caches only
`["Z","RA","DEC","WEIGHT"]`. Extend `cols` to include
`WEIGHT_COMP, WEIGHT_SYS, WEIGHT_ZFAIL, PROB_OBS, NTILE` (data *and* randoms)
before starting — every test below needs them, and the re-read is a few
minutes once.

### T1 — α(NTILE) flatness  *(minutes; do this first)*

If the completeness correction is applied consistently between data and
randoms, then `α(NTILE) = Σw_data(NTILE) / Σw_rand(NTILE)` must be flat, equal
to the global α = 0.105 up to clustering scatter.

```python
for n in range(1, 8):
    m_d, m_r = ntile_d == n, ntile_r == n
    print(n, w_d[m_d].sum() / w_r[m_r].sum(), m_d.sum())
```

This is a direct test of Eq. 8.2 consistency and of `SYSTEMATICS.md` trap #7.
**Decision:** flat to <1% → item 3 of §2 is closed. Any systematic trend with
NTILE is a bug in the weighting, must be fixed before anything else, and would
by itself explain a low-ℓ deficit.

### T2 — NTILE-split consistency  *(hours; the strongest data-only test)*

This is the closest data analogue of complete-vs-ALTMTL that exists without
mocks. Split into a high-completeness subsample (NTILE ≥ 4, C_assign ≈ 72–92%)
and a low one (NTILE ≤ 2, C_assign ≈ 35–48%), with **matched per-subsample
randoms**, and measure `Cl^{g,CIB}` in each. The high-NTILE region *is* the
near-complete sample.

If the weights work, the two amplitudes agree. If the §5 mechanism is operating,
the low-NTILE amplitude is suppressed, and suppressed more at high ℓ.

Controls that must be included or the test is worthless: the two subsamples sit
on different sky, so they see different CIB, different depth, different dust.
Run the identical split on the **randoms-only** and on a **CIB-rotated null** to
calibrate how much difference the footprint change alone produces. Quote the
comparison as a ratio to the theory prediction for each subsample's own n(z),
not as a raw amplitude difference.

**Decision:** agreement within ~1σ → strong evidence the weights are enough,
and this becomes the paper's fiber null test. A completeness-scaling deficit →
you have found the residual on data and T5 becomes mandatory.

### T3 — weight-variant spread  *(hours)*

Measure `Cl^{g,CIB}` in the fiducial bin with four galaxy weightings:
`WEIGHT` (fiducial) / `WEIGHT` with `WEIGHT_COMP` divided out /
`1/PROB_OBS` substituted for `WEIGHT_COMP` / uniform. Randoms handled
consistently in each case.

This gives you two numbers you currently lack: **the size of the correction**
(fiducial vs no-comp) and **the sensitivity to how it is defined** (IIP vs
`f_TLID·f_tile`). It is the same logic DESI uses for its own weight-scheme
systematics.

**Decision:** if the correction itself is only a few percent, then even a 20%
error in the weights is sub-percent and the whole question closes on an
error-budget argument alone — which is a perfectly rigorous outcome and the
cheapest possible win. If the correction is 20%+, weight accuracy matters
linearly and you need T2/T5.

### T4 — additive term: deproject the completeness template  *(hours)*

You have already measured the completeness × CIB correlation. Turn it into
either a correction or a bound: build `c(n̂)` from the per-bin random density,
pass it to `NmtFieldCatalogClustering(..., templates=[c], lmax_deproj=...)`,
and compare deprojected vs fiducial `Cl^{g,CIB}`.

**Decision:** shift ≪ σ → quote as a bound and move on. Shift comparable to σ →
adopt deprojection as fiducial and report both. Cheaper and better targeted
than any θ-cut analogue, as `SYSTEMATICS.md` §4 already notes.

### T5 — `T(ℓ)` from κ mocks  *(needs a collaborator; the decisive test)*

**First ask what the third mock version was** (§5). The suite is Y3 and the
footprint is matched, so if the third version is weighted-ALTMTL then `T(ℓ)`
already exists and this collapses to a re-plot of `weighted-ALTMTL / complete`
instead of a re-run.

Failing that, request precisely: **weighted ALTMTL vs complete `Cl_kg`, per
dz=0.2 tomographic bin, matched randoms per realization, identical κ map/mask/
MCM on both sides, ratio and its scatter across realizations.** Ask explicitly
which randoms were used on the ALTMTL side — target-selection randoms or
per-realization fiber-assigned randoms — since that alone can generate the
low-ℓ part of the deficit.

**Decision:** `T(ℓ)` consistent with 1 → the fiber question is closed with a
measurement rather than a citation, and that is what goes in the paper.
`T(ℓ)` ≠ 1 → divide it out of the data (or fold it into the model as a
transfer function, with its mock scatter as an error term) and quantify the
z-dependence across bins.

### T6 — `p_obs` estimator-noise bias  *(hours; only if T3 shows a large correction)*

If `BITWEIGHTS` are available, recompute `p̂` from subsets of N=32, 64, 128
realizations and watch `Cl^{g,CIB}` move. Extrapolating in 1/N isolates the
Jensen bias of §2 item 2 directly. If only `PROB_OBS` is on disk, bound it
analytically from the `p` distribution instead: `bias ≈ σ_p²/p²` per object,
weighted by the density field.

---

## 8. Recommendation

**Do not close this on the literature argument.** [2505.20656] §3.2 is a
correct statement of mechanism made at a precision an order of magnitude looser
than yours, with weights cruder than yours, and with no supporting measurement.
Citing it is right; resting on it is not, and a referee who has seen the
ALTMTL-vs-complete figures will say so.

**Do not build a θ-cut analogue.** §3. It is the right call and it should be
argued positively in the paper — that the estimator's linearity in `δ_g` is
what removes the need, and that the price of that linearity is that the weights
carry the whole load.

**Sequence:** step zero (§5) in parallel with T1 → T3 → T2 → T4, then T5 only
if T2/T3 indicate it is needed or if step zero reveals the mock deficit is
post-weight. T1 and T3 together are perhaps a day and can plausibly close the
question by error budget.

**Revise `SYSTEMATICS.md` §4.** Three specific edits:
- Change the heading from "Settled: fiber assignment" to reflect that (b) is
  settled and (a) is argued.
- Correct the framing that the residual sits at ℓ≈3600. The patrol-radius
  effect does; the weight-failure modes do not, and they land at low ℓ (§2).
- Note that the residual-risk table (comp×CIB) is the *additive* channel only
  and does not bound the multiplicative one.

---

## 9. Adjacent risk worth flagging now — imaging weights × CIB

Out of scope today but it will bite this measurement harder than fiber does,
and it is on the project's systematics list.

[2411.12020] §6.4: ELG imaging weights come from a **SYSNet neural-net
regression against all validation maps except `EBVnoCIB`** — which explicitly
**includes `HI` column density**. The Lenz19 CIB maps are constructed *by using
HI column density to remove Galactic dust*. So the ELG selection function has
been regressed against a map that is a direct ingredient of your CIB map. That
is a clean channel for the imaging weights to either remove real
galaxy–CIB signal or imprint a spurious one.

DESI has already seen this failure mode in the BGS sample and says so plainly
(§6.1): the ΔEBV GR trends are "primarily driven by CIB contamination in SFD
E(B−V)" and "any regression applied that would remove the trends is likely to
remove real clustering modes (traced by CIB)."

Concrete first step, and it is the same shape as T3: measure `Cl^{g,CIB}` with
and without `WEIGHT_SYS`. `SYSTEMATICS.md` §6 item 1 already flags
`WEIGHT_SYS` ∈ [0.69, 2.0] as untested; this makes it the higher priority of the
two open systematics, not the lower.

---

## References

- [arXiv:2411.12020](https://arxiv.org/abs/2411.12020) — DESI 2024 II. §5.1–5.3
  completeness definitions and `w_comp`; §5.2 Eq. 5.1 `p_obs` from 128 altMTL
  realizations; §6.4 ELG SYSNet map list; §6.1 SFD/CIB contamination; §8.2
  Eq. 8.1–8.4 `w_tot`, the `<w_comp>(n_tile)` refactoring, `P_0,ELG = 4000`;
  Table 6 completeness and area vs NTILE.
- [arXiv:2406.04804](https://arxiv.org/abs/2406.04804) — θ-cut estimator and
  window. Λ_θ = 0.05°; DR1 ELG ~35% complete; ELG fiber-assigned/complete pair
  ratio drops to ~20% at small θ in SGC.
- [arXiv:2411.12025](https://arxiv.org/abs/2411.12025) — fiber assignment
  characterization. Weighting schemes agree above 20 h⁻¹Mpc; up to 60%
  suppression for ELGs below θ ≈ 0.02°; ~2.5×10⁵ clustered zero-probability ELG
  targets in NGC.
- [arXiv:2505.20656](https://arxiv.org/abs/2505.20656) — DESI DR1 galaxy × CMB
  lensing. §3.2 and Table 2; the "completeness weights are adequate for
  `C_ℓ^κg`" and "PIP has no meaning for `C_ℓ^κg`" statements, and fn. 9 on IIP.
- [ApJ, CIB tomography with SDSS/BOSS/eBOSS](https://iopscience.iop.org/article/10.3847/1538-4357/adfb6a)
  — closest spec-z × CIB precedent; real-space `w(θ)`, does not treat fiber
  collisions.
