# THz-TDS Analysis — Decisions & Methodology Notes

Living record of the key analysis decisions, the physics behind them, and the
gotchas we have hit. Intended to seed a tutorial / comprehensive docs later.
Newest material appended at the bottom of each section.

---

## 1. Geometries

### Transmission (free-standing slab)
- `H = Y_sample / Y_reference`; reference is the open beam (or substrate).
- Sample pulse arrives *later* than the reference by the propagation delay
  `(n-1)·d/c` — this linear phase **is the signal** (it gives n). Do **not**
  remove it by timing alignment (see §4).
- Invert with `invert_nk(f, H, thickness_m, mask)` — needs the sample thickness.

### Reflection — window-coupled (SiO₂ window, sample pressed on back)
- 45° external incidence, s-pol. One acquisition holds the **first reflection**
  (air→SiO₂ front face) and the **second reflection** (SiO₂→sample back face).
- The beam refracts into the window: **internal angle ≈ 21.3°** (Snell,
  n_SiO₂≈1.95), *not* 45°. The back-interface Fresnel inversion uses
  `n_incident = n_SiO₂` at the internal angle.
- **Computed back-face reference:** the SiO₂-only second reflection is
  `r_{SiO₂→air}`, which cancels the window path when ratioed; absolute scale is
  restored by the *computed* Fresnel `r_{SiO₂→air}`. So
  `r_sample = r_{SiO₂→air} · H` with `H = second_ref_sample / second_ref_SiO₂only`.
- Vocabulary: "echo" is reserved for the **GaP detection-crystal echo**
  (~+5.3 ps replica of the main pulse, instrumental). The sample-bearing pulse
  is the **second reflection**, never called an echo.

### Reflection — gold-referenced (flat sample in air)
- Flat sample reflects in air at 45° s-pol, referenced to a gold mirror
  (`r_gold ≈ -1`). `r_sample = -H`, `n_incident = 1`, incidence angle = 45°.
- Used for the silicon benchmark (CNT-4 `Si2`/`Si2-90`). No window, single
  reflection pulse, so no segmentation needed.

### Sign convention
- Reflected pulse **inverts** (r<0, π flip) off a **higher**-index medium.
  SiO₂→Si (n_Si≈3.4): inverts. SiO₂→air: does not. Magnitudes can be similar,
  so the **sign/phase is the discriminator**, not the amplitude.
- Diagnostic for contact: overlay sample vs bare-window second reflection.
  Opposite polarity ⇒ seeing the high-index sample; same polarity ⇒ air gap.

---

## 2. Segmentation (reflection only)

- `segment_reflections` is a **clean splitter**: it crops each raw multi-scan
  trace to each component's time gate and writes a `.acc` per component into
  `segmented/<component>/<original_name>.acc`. **No zero-pad, no taper, no
  in-memory processing.** Reload and process normally.
- Per-component subfolders keep original filenames ⇒ grouping/pairing on the
  reloaded files is identical to the originals.
- Gates exclude the GaP echo. Keep the second-reflection gate trailing edge
  below the *next* GaP echo of the second reflection.
- **GaP second echo exclusion (CNT-17, 2026-06-16).** A *second-order* internal-reflection
  echo inside the GaP detection crystal (distinct from the first ~+5.3 ps replica)
  sits ~168 ps and adds interpretation-harming oscillations. **No dedicated crop is
  needed**: it sits at a fixed delay from T0, so it is excluded by the second-reflection
  region's trailing edge (e.g. 168 ps), and `isolate_and_window` zeros everything
  outside each region before windowing — so even a `pad`-mode window reaching past the
  region edge multiplies zeros there (verified: max |amp| beyond 168 ps = 0). A brief
  `build_full_trace_reflection(hard_crop_ps=...)` arg was removed as CNT-17-specific
  scaffolding. Consequence — the second reflection (peak ~165.4 ps) sets trusted-band n
  (§16) yet excluding the echo leaves it only ~2.6 ps post-room → narrow symmetric
  window (~0.19 THz res); the fix is on the acquisition side. See [[gap-second-echo-crop]].

---

## 3. eps_background for conductivity  ★ key for doped semiconductors

- `sigma = -i·ω·ε₀·(ε − eps_background)`.
- **Free-standing / vacuum:** `eps_background = 1`.
- **Doped semiconductor (e.g. doped Si), to isolate the free-carrier Drude
  response:** set `eps_background = ε_lattice`. For silicon, **11.7**
  (= n_lattice² ≈ 3.418²).
- Why: with background = 1, the silicon *lattice* polarization (ε≈11.7)
  dominates σ — σ₂ becomes large, negative and monotonic with **no σ₁/σ₂
  crossing**. This is the "conductivity shape looks wrong / imaginary off"
  symptom. With background = 11.7 the lattice is removed and the free-carrier
  conductivity (with its σ₁/σ₂ crossing near ωτ=1) is isolated.
- **Decision (2026-06-07): use `eps_background = 11.7` for the doped-Si work.**

---

## 4. T0 calibration / phase  ★ critical for reflection

- A bulk timing offset between sample and reference appears as a linear phase
  ramp `exp(-iωΔt)` in H. In **reflection** there is *no* propagation term, so
  this linear phase is dominated by the **instrument** (stage drift, window
  bowing) — removing it is safe and necessary. In **transmission** the linear
  phase *is* the signal `(n-1)d/c` — do **not** remove it.
- **Method chosen: cross-correlation** (`align_to_reference`, wraps thz-core
  `align_time`). Anchors the reference, shifts the sample, sub-sample (parabolic)
  precision, uses `|corr|` so it handles the sign-flipped sample pulse.
  - Rejected `centering_manual`: integer-sample shifts only (0.05 ps granularity) —
    too coarse for a ~0.1 ps offset.
  - Rejected phase-slope removal as the primary tool: equivalent in principle,
    but needs a trusted band and over-corrects in practice (tested: it made the
    Si benchmark *worse*).
- Run in the processing phase **before** window_time/FFT.
- Residual after alignment is *not* a pure slope (removing the leftover slope
  hurt), so further timing removal is not the fix for residual k.
- **Sub-sample shift handling (2026-06-08).** `align_to_reference` no longer
  interpolates the time-domain Y (interpolation is a lossy low-pass). Instead it
  splits the measured shift: the **whole-sample** part slides the time axis
  (exact, picked up by `pad_to_common_grid` as an integer offset); the
  **sub-sample residual** (|.| ≤ dt/2) is applied later in `transfer_function`
  as an **exact spectral phase ramp** `H·exp(-iω·Δt_residual)` (Fourier shift
  theorem = exact sinc interpolation, no smoothing). Leaving the residual in the
  axis instead would let `pad_to_common_grid`'s `int(round(...))` silently drop
  it, leaving a linear phase error ω·δt that biases n worst at high f
  (≈π·f·dt per half-sample; ~0.47 rad at 3 THz for dt=0.05 ps).
  - **Toggle:** module constant `thz_adapter.SUBSAMPLE_TIMING_CORRECTION` (single
    line), or per-call `align_to_reference(..., subsample_correction=False)`.
    Both align and transfer print the split + whether the ramp was applied, so
    it is obvious when it is in use. Disable to A/B test its effect.

---

## 5. Assessment methodology  ★

- **Assess the frequency-dependent shape, not a band-average.** Band-means catch
  gross failures (e.g. n jumping 2.31→3.35) but hide (a) low-frequency
  breakdown and (b) genuine dispersion vs noise. A reported sd over a band
  conflates dispersion with noise and is misleading as an error bar.
- For a conductor, judge by: σ₁(f), σ₂(f) curves, the **σ₁/σ₂ crossing
  frequency** (≈ 1/2πτ for Drude), and a **Drude / Drude–Smith fit**
  (thz-core fitting registry) reporting σ_dc and τ.
- **Mask the low-frequency breakdown.** With short acquisition windows the band
  edges (here <~0.5 THz) blow up (n→1.5, k spikes) from low SNR + coarse bins;
  exclude them from fits.
- **Window length matters:** a 7 ps acquisition ⇒ ~0.14 THz bins ⇒ noisy shape.
  Longer scans materially improve the extracted shape.

---

## 6. Silicon benchmark — status (2026-06-07)

- Data is good: |r_Si|/|r_gold| ≈ 0.66 matches n≈3.4 at 45° s-pol (Fresnel 0.649).
- Without alignment: n≈2.31, k≈1.52 (wrong — phase misread as loss).
- With `align_to_reference`: `Si2` n≈3.35, `Si2-90` n≈3.37 (consistent; same
  isotropic wafer at 0°/90°), near literature n=3.418. Reflection geometry
  ='gold', 45°, s-pol.
- Open issues: n scatters 3.0–3.8 above 0.6 THz, k≈0.1–0.3 (should be ~0 for the
  lattice), Drude σ₁/σ₂ crossing expected <1 THz not yet clean. Cross-checking
  against transmission (Chris d≈330 µm; CNT-4 `Si-t3` is a transmission meast.)
  to separate code vs sample/measurement effects.
- `Si-t3` is a **transmission** measurement (explains its anomalous −1.75 ps
  "reflection" shift); use it as a transmission cross-check, not a reflection.
- Prior `si2_results.csv` was all-NaN; `Chris/results/*.csv` had n~1e9
  (thickness/units blow-up). Both are bad outputs, not ground truth.

### Transmission vs reflection cross-check (2026-06-07)
- **Transmission (Chris, doped Si):** after two fixes — (a) anchor the phase
  unwrap at low frequency, (b) `eps_background=11.7` — the conductivity is a
  **clean textbook Drude**: σ₁ rolls off (1.95→0.9 S/cm), σ₂ rises (0.95→2.4),
  crossing at **0.96 THz** (matches the expected <1 THz). n flat ≈3.02 at
  d=330 µm (⇒ true d≈281 µm if n=3.418), k smooth Drude rolloff 1.2→0.13.
- **Reflection (Si2):** n≈3.35 (correct) but σ is **noisy/scattered** (multiple
  spurious crossings 1.4–1.7 THz, σ₁ dips toward 0). NOT a clean Drude.
- **Same-wafer test (2026-06-07):** `Si-t3` is the single-pass transmission of
  the *same* wafer as the `Si2` reflection (d=350 µm; raw 2.80 ps delay ⇒
  n≈3.40, confirming gold is a valid transmission reference). Transmission:
  n≈3.29, k≈0.24, **smooth σ, single clean crossing at 1.18 THz**. Reflection
  (same wafer): n≈3.49 (agrees ~6%), σ noisy with spurious crossings 1.4–1.7 THz.
  ⇒ On the identical wafer, reflection gets **n** right but σ is unreliable —
  the definitive proof that reflection is a weak σ probe for low-loss Si.
- **Conclusion:** the two geometries **agree on n** within the thickness
  uncertainty (code is consistent); the σ difference is *not* a reflection-code
  bug per se. The reflection σ is noisy because of: (i) short 7 ps window
  (~0.14 THz bins), (ii) residual phase, and (iii) **a fundamental sensitivity
  limit** — for a low-loss high-index sample (Si: n≈3.4, k≈0.3) the small loss
  barely perturbs a single-interface reflection, so reflection is a *weak probe
  of σ*. Transmission accumulates absorption over the path length and measures σ
  far better. Highly-conductive samples (CNT, large k) are the opposite — well
  suited to reflection. So **Si validates reflection n, but is a poor σ
  benchmark for reflection**; validate reflection σ with a conductive standard
  or by comparing CNT reflection vs CNT transmission.

## 6b. Reflection conditioning — window vs air/gold for conductors ★ (2026-06-07)
- Inverting `r` for n,k has `(1+r)` in the denominator, so it is **ill-conditioned
  as r → −1** (a near-perfect reflector). Conductive samples (CNT, metals) push
  r toward −1, so the inversion can blow up.
- **The incident-medium index sets the conditioning.** Near r=−1,
  `|1+r| ≈ 2·N_incident / |N_sample|`. So a higher incident index (SiO₂, 1.95)
  keeps r further from −1 than air (1.0) for the same sample — roughly a factor
  N_incident better conditioned.
- **Measured (CNT):** air/gold reflection (N_inc=1) blows up — n→120,
  σ₂→−12500 S/cm at 1.5–1.7 THz, where the CNT reflects ≈ as well as gold so
  r≈−1. SiO₂-window reflection (N_inc=1.95) of CNT is stable and bounded
  (n≈4.5 flat, k≈0.5–1, σ₁≈1–10 S/cm). n agrees (~4.7) where both are stable.
- **Consequence:** the SiO₂ window is not just a contact/alignment convenience —
  it **fundamentally improves the inversion for conductive samples** by lowering
  the reflection contrast. Air/gold reflection is a poor geometry for highly
  conductive films (same "indium problem": r≈−1, σ-insensitive, ill-conditioned).
- Caveats: different CNT samples in the two sets (not same-sample); window σ₂ is
  negative/growing — likely needs a CNT `eps_background`>1 (bound π-electron
  permittivity), not the σ; single-interface inversion assumes a semi-infinite
  sample, but CNT paper is a finite film (absolute σ is an effective value).

## 6c. Air/gold CNT-4 measurement caveats (someone else's data)
- Done in air, no window, "near 45°" — angle was set by overlapping the CNT T0
  with the gold T0 (CNTs don't back-reflect the NIR alignment laser). The sample
  surface sat a few hundred µm off the gold plane.
- Aligning T0 by **angle** (not by translating along the normal) ⇒ the CNT is at
  an unknown angle ≠ gold's, plus **defocus** from the displacement ⇒ systematic
  angle bias + frequency-dependent coupling loss. `align_to_reference` finds ~0
  shift (they pre-matched T0), so the data looks aligned but the geometry error
  is baked in. Combined with the ill-conditioning above, the air/gold CNT n,k,σ
  are unreliable.

## 8. Kramers-Kronig phase correction (Jatkar et al. 2024) — prototype assessment (2026-06-08)
- Standalone module `thz-core/thz_core/kramers_kronig.py` (`estimate_misplacement`,
  `correct_reflection_phase`); tests `tests/test_kramers_kronig.py`. Method:
  misplacement adds a linear-in-ω phase; amplitude is robust; inverse-KK + an
  analytical fit of `Δ_m(ω)` recovers the misplacement and the true phase.
- **Window length is NOT the binding constraint.** Time-window length sets
  frequency RESOLUTION; KK is insensitive to it (grid-density sweep: l recovery
  invariant 400→6400 pts). KK needs BANDWIDTH — `f_end` comfortably above the
  spectral feature — and bandwidth is set by pulse/detector SNR, not window
  length. So the echo-limited short window does not directly cripple KK; zero-pad
  for a fine grid and the real question is usable bandwidth vs feature location.
- **Binding constraint = truncation bias.** The intrinsic-phase tail above
  `f_end` (paper's `E_ωend`) is not perfectly constant ⇒ systematic ~15% l
  under-estimate (prototype), NOT fixed by finer grid. Bias GROWS as `f_end`→
  feature: in tests l collapses (−34→−4 µm) as `f_end` drops to the feature.
  Rule of thumb: KK reliable when `f_end ≳ 1.5–2× feature frequency`.
- **Our collimated beam is an advantage** (no defocus on displacement → clean
  linear phase even for large shifts; the paper's focus-artifact limit doesn't
  apply). `φ_0`=0 for metals; window CNT's π sign-flip needs `φ_0`=π handled.
- **Verdict for us:** KK is correct-in-principle and a good cross-check, but in
  our likely bandwidth-limited regime it carries a truncation bias and is
  fit-band sensitive, so it is NOT a clear win over cross-correlation (which is
  precise/simple but assumes a pure time shift). Prototype accuracy ~15–25%
  (below the paper's sub-µm; their accuracy needs better PV integration / fit
  tuning or their Method-2 minimisation). Recommend: keep cross-correlation as
  primary; use KK as an independent cross-check; revisit if we gain bandwidth.

## 7. Phase-branch in transmission inversion  ★ FIXED 2026-06-07
- **Bug:** `invert_nk` unwrapped the phase over only the **analysis mask**. If
  the mask started too high (where |φ|>π for a thick sample), `np.unwrap` could
  not recover the wraps below its first sample, so n was wrong by a
  frequency-dependent amount (Chris: masking from 0.4 THz → n=2.18; from 0.1 THz
  → n=3.02). The integer `n_offset_2pi` knob can't fix it cleanly (~2.3 per step).
- **Fix:** unwrap the contiguous span from the lowest finite positive-frequency
  bin up to the top of the mask, then restrict to the masked bins. Low-THz has
  |φ|<π and strong pulse content, so the branch anchors reliably. Also handles
  gapped masks. Tests: `tests/test_invert_unwrap.py` (3). Verified on Chris:
  n=3.020 now regardless of mask start (was 2.18 from 0.4 THz).
- Reflection inversion never had this problem (no propagation phase; n comes
  straight from the Fresnel coefficient) — which is why Si2 reflection n was
  robust while Chris transmission n needed the fix.
- The legacy `silicon_test.csv` n~1e9 was likely a severe version of this bug.

### 7b. SNR-guided unwrap — `robust_unwrap` (2026-06-09)
- **Motivation:** §7 fixes the *thick-sample* wrap-count (where to start the span).
  A second failure is *noise*: a run of low-SNR bins whose phase random-walks
  across ±π gets integrated by `np.unwrap` into a net spurious wrap that offsets
  every later bin by 2π.
- **Key insight (non-obvious, drove the whole design):** "just start the unwrap at
  a high-SNR bin" does **not** help. `np.unwrap` = cumsum of principal-value
  differences between *consecutive* samples; each ±2π decision is step-local and
  direction-independent, so starting the walk elsewhere crosses the same steps and
  decides them identically. (A first bidirectional-from-anchor implementation was
  provably equal to `np.unwrap` up to a constant — no robustness, and it broke the
  absolute branch.) Also: a *single* outlier only spikes (its two steps cancel) —
  it is *runs* that persist.
- **What works:** `thz_core/unwrap.py::robust_unwrap` *excludes* low-SNR bins from
  the wrap decisions — unwrap across trusted bins only, then snap the rest onto
  that branch. `weights=None` ⇒ byte-for-byte `np.unwrap` (robustness is opt-in);
  `snr_floor` drops a low-amplitude tail.
- **Caveat:** excluding low-f bins discards the absolute wrap count, so it is only
  safe for reflection/thin samples (|φ|<π, no real wraps). Thick transmission must
  keep its low-f bins trusted. Shared by `invert_nk(snr_weights=)` and the KK
  estimator. Tests: `test_robust_unwrap.py` (8) + an `invert_nk` noise regression.
- **Follow-up (open):** thread a real SNR proxy (reference |spectrum| / dynamic
  range from `trusted_band_mask`) through the adapter so it is on by default.

## 9. Sample contact gap (window reflection)  ★ conceptual / not yet implemented (2026-06-09)
Context: re-pressed CNT-on-glass under gentle force still showed a T0 shift of
0.155–0.18 ps, changing with 90° rotation. Question raised: is this an intrinsic
reflection delay, or a gap?

- **It is geometric, not intrinsic.** Intrinsic reflection group delay (`τ=−dφ/dω`)
  is tiny: a lossless dielectric interface is real & frequency-flat ⇒ τ≈0 (a π flip
  is a sign, not a delay); a metal is ~skin-depth/c ≈ sub-fs; a Drude/CNT conductor
  is at most ~its scattering time (tens of fs) **and frequency-dependent** (phase
  curvature, not a flat ramp). None give a flat 0.18 ps shift. Glass→metal→CNT
  ladder: ≈0 → ≈0 → tens of fs.
- **0.18 ps ⇒ a ~35–40 µm air gap.** `Δt = 2·n_gap·d·cosθ_gap/c`; by Snell the gap
  angle is 45° (cos=0.707): 0.18 ps→38 µm, 0.155 ps→33 µm, the rotation difference
  ⇒ ~5 µm change. Plausible for a rough CNT mat under gentle force. Diagnostic: the
  two reflections (window face + delayed CNT) are closer than the pulse width, so
  they interfere into one *shifted* feature rather than resolving as two pulses.
- **The gap negates the window's purpose.** With a gap the CNT sees air (n=1), not
  SiO₂ (n=1.95), so we lose the incident-index conditioning advantage (§6b) *and*
  add an etalon. Contact quality is the whole ballgame.
- **Exact de-embedding (the right "informed background removal").** Naive
  subtraction is wrong (it ignores the etalon's multiple bounces). The gap is a
  Fabry–Pérot layer: `r_meas = (r1 + r2 e^{−2iβ})/(1 + r1 r2 e^{−2iβ})`,
  `r1 = r_{SiO₂→air}` (known/measured), `β = (ω/c)·d·cosθ_gap`. Solve exactly for
  the CNT term: **`x = r2·e^{−2iβ} = (r_meas − r1)/(1 − r1·r_meas)`**. Well
  conditioned (denominator ≥0.56). Then `|x| = |r2|` exactly (lossless gap) and
  `arg(x) = arg(r2) − 2β` where 2β is a *pure linear phase* ⇒ removed by the
  existing T0/cross-correlation step. So for a *smooth single-valued* gap the
  window-face reflection and the etalon are removable in closed form.
- **Roughness is the binding limit (frequency-dependent).** A spot illuminates a
  *distribution* of gap d. The phase spread per gap variation is `2·(2π/λ)·cosθ`;
  for ±20 µm roughness that's ≈±0.6 rad at 1 THz (tolerable) but ≈±1.8 rad (~π) by
  3 THz (coherent reflection washes out). So roughness is a Debye–Waller-like
  high-pass damping on |r2| → **usable bandwidth ends where λ approaches the
  roughness scale**. The exact de-embedding holds at low f and degrades at high f;
  its |x| roll-off is itself a free coherence-bandwidth diagnostic.
- **Pressure series** is analytically usable: parametrise each run by its measured
  gap (from the delay) and either extrapolate r2 to d→0, or do a global fit with
  shared r2(ω), per-run d, and an optional fitted roughness σ_d.
- **Masking-a-small-patch idea:** the useful kernel is "a small patch presses
  flatter" — so make the patch *larger than the beam* (no window component at all).
  Progressive-aperture spatial unmixing is workable in principle but finicky at THz
  (diffraction at aperture edges within a few λ); the algebraic de-embedding removes
  the window contribution without touching the beam.
- **Recommended order:** (1) better mechanical contact (small flat patch, firm
  uniform pressure — restores §6b conditioning); (2) implement the de-embedding step
  (exact for the smooth part, free roughness diagnostic); (3) pressure-series global
  fit for quantitative n,k despite imperfect contact.
- **Open / offered:** prototype the de-embedding + a synthetic test (smooth gap →
  exact r2 recovery; Gaussian d-distribution → high-f |x| roll-off).
- **Prototype done (2026-06-18, `explorations/explore_air_gap_deembedding.py`,
  isolated demo — not pipeline code).** Synthetic Drude CNT behind a known gap.
  Confirms: (1) `x = (r_meas − r1)/(1 − r1·r_meas)` recovers `|x|=|r2|` to 1e-16;
  (2) stripping `2β` with the **true** gap d recovers n,k to **4e-15 (exact)** — the
  algebra is sound; (3) the **practical weak link is the gap-thickness estimate**: a
  naive linear fit of `arg(x)` overestimates d (34 vs 30 µm here) because the sample's
  **own** dispersion adds phase slope, leaving a residual that mis-scales n (~1.1 error).
  So d must come from the **pulse round-trip delay** (T0/cross-correlation of the
  second reflections), not from `arg(x)`; or iterate (estimate d → invert → model
  `arg(r2)` → subtract → re-fit). (4) Roughness σ_d=12 µm rolls `|x|/|r2|` below 0.5 by
  ~3.5 THz — the coherence-bandwidth ceiling. Naive (no de-embed) n drops to 0.76 by
  3 THz, reproducing the CNT symptom (§9b). Next: a real-data application (de-embed the
  CNT-17 `r_sample`, d from its measured second-reflection delay) before any pipeline
  integration.

### §9b  Reflection phase-code audit — the n<1 droop is the gap, not a code bug (2026-06-18)
Samuel saw CNT reflection `n` fall **below 1 at high frequency** and not fit a model,
suspected the air gap, and asked whether the reflection phase code had a hidden bug
like the transmission one (§18). Audit (`explorations/explore_reflection_phase_audit.py`,
CNT-17 second_reflection self-ref, 5 samples/rotations):
- **Reflection inversion is immune to the §18 bug.** `invert_nk_reflection` is
  closed-form on the **complex** `r = r_ref·H` — no `np.unwrap`, no DC anchor, no
  re-unwrap. An integer 2π (or the constant offset that bit transmission) is invisible
  to it (`exp(i·2πm)=1`) and is used directly, so there is no branch/cycle to lose or
  mishandle. The transmission failure cannot occur here.
- **φ(H) carries a consistent LINEAR phase** of +0.107…+0.127 ps across all five
  samples and rotations (s-0/45/90/180) ⇒ a 16–19 µm equivalent air gap (`d=c·Δt/2`).
  Consistent magnitude across rotations ⇒ systematic gap, **instrumental** — exactly
  the §9 contact gap, confirming Samuel's suspicion. A linear phase rotates `r` with
  frequency; in the closed-form inversion that pushes `n` down monotonically and
  through 1 at high f (raw n: ~2.4 @0.7 THz → <1 above ~1.1–1.5 THz). De-embedding the
  fitted linear phase reverses the droop (a crude full-band fit overshoots to n≈6–28,
  confirming the *mechanism*; the exact §9 Fabry–Pérot de-embed is the correct removal).
- **φ(H) intercept ≈ π (mod 2π)** for every sample. This is **physical, not a bug**:
  the CNT is higher-index than the SiO₂ window, so `r_{SiO₂→CNT}` is negative (phase π)
  while the reference `r_{SiO₂→air}` is real-positive ⇒ `H` is negative-real ⇒ arg≈π.
  The closed form handles it natively (a sign, not a delay — cf. §9 first bullet).
- **Verdict:** the reflection phase treatment is sound; the n<1 droop is the air-gap
  linear phase (instrumental). The fix is the §9 Fabry–Pérot de-embed (designed, not
  yet implemented), not a phase-code change. Note the self-ref path skips the sub-sample
  ramp by design (front-pulse ratio carries timing), so no spurious code timing is added.

## 10. Artificial time-shift sweep (qualitative phase exploration)  (2026-06-09)
Tool to scrub an artificial sub-sample T0 shift on a chosen sample and watch where
n/k/σ land — for testing "which shift reproduces the expected (Drude-like) shape".
- **Implementation:** `thz_adapter.sweep_time_shift` (array result),
  `time_shift_slider` (blitting interactive), `time_shift_waterfall` (static map).
  The shift is applied EXACTLY as a spectral phase ramp `H·exp(-i·2πf·Δt)`
  (`core.phase_ramp`) — **no time-domain interpolation** (which would low-pass the
  very shape being assessed). Each step is exact to float precision; sub-sample
  steps are arbitrary-fine. No integer/fractional split needed here (the split in
  `align_to_reference` is only to keep the *plotted* trace visually aligned).
- **Cheap by construction:** |H| and the SNR mask are invariant under a phase ramp,
  so they're computed once; the slider pre-sweeps the whole range, then each frame
  is an array lookup + `set_ydata` + blit. Reflection inversion is closed-form
  (no iteration) → per-frame cost is negligible; rendering is the only limiter.
- **★ Methodological caveat (confirmation bias).** Sweeping Δt is manually
  exercising the phase-calibration degree of freedom; you *can* dial n over a wide
  range and coax a Drude-looking σ at more than one shift. "It matches at Δt=X" is a
  hypothesis, not proof. Honest cross-check: the Δt↔n mapping is closed-form (a
  shift adds a known linear phase; the window inversion is analytic), so
  back-calculate the Δt that *should* give the expected n at a reference frequency
  and confirm it agrees with the slider landing — if not, the "expected shape" came
  from somewhere the shift can't legitimately reach.
- **Display scaling ★ (gotcha).** The reflection inversion spikes 1–2 orders of
  magnitude at low-SNR band edges (the `|1+r|→0` conditioning limit, §6b): for
  CNT-10A `a-12.6_cnt-s`, σ₁ median ≈74 but 1200–1644 at 3.5–4.2 THz. Auto-scaling
  to min/max then flattens the real sub-1 THz structure to an invisible line/flat
  colour (this read as "no data" / "nothing above 0.5 THz"). Fix: robust
  percentile limits — slider y-limits 1–99 %, waterfall colour 2–98 %
  (`_robust_limits`). `band_thz=` overrides the SNR mask for exploration past the
  trusted band (data there is real but below SNR — interpret accordingly).
- **Slider rendering:** uses `set_ydata` + `draw_idle` (not manual blitting, which
  fought the Slider widget and left the curve undrawn); fast enough for the
  few-hundred-point curves. An initial `update()` draws the curve immediately.
- Tests: `tests/test_time_shift_sweep.py` (6); `test_phase_ramp.py` (2, thz-core).

## 11. Front-pulse self-referencing & window characterisation  ★ implemented (2026-06-11)
Problem: the bare-window reference and the pressed-sample measurement are
different mounts. Window deformation under pressure / rotation / realignment
changes the coupling into the spectrometer, so the reference no longer matches
the sample's window path — frequency-structured "fake features" appear in H.

- **Key identity.** The intra-trace ratio `W(ω) = Y₂/Y₁` of a bare-window trace
  is a property of the window alone: source spectrum, detector response and the
  shared air path are common to both pulses and cancel. Model (s-pol):
  `W = [t_in·r_back·t_out/r_front]·exp(−2i(ω/c)·d·β)`, `β = √(n_w²−sin²θ_e)`.
  The Fresnel factor is ≈ −0.81 for SiO₂ at 45° (sign flip between pulses).
- **The correction.** `H_new = (Y₂ₛ/Y₁ₛ)/(Y₂ᵣ/Y₁ᵣ) = H_old·(Y₁ᵣ/Y₁ₛ)` — a ratio
  of intra-trace ratios; each trace is referenced to its own front pulse, which
  never sees the sample. The front-pulse ratio `D = Y₁ₛ/Y₁ᵣ` IS the drift, and
  is directly measurable (D ≡ 1 would mean no drift).
- **Evaluation on CNT-13/D (2026-06-10).** Mount-to-mount drift |D|−1 =
  6.6–9.6% rms with oscillatory structure (swings 0.93–1.22). Leave-one-scan-out
  CV on the bare window: predicting Y₂ from Y₁ via W has a 1.6% rms noise floor
  (vs 1.0% direct substitution — the ~√2 cost of using two noisy pulses). So the
  correction trades a 7–10% structured systematic for ~0.6% extra noise.
  Full-pipeline result: pairwise spread across the three s-orientation repeats
  drops n 11.3%→4.1%, σ₁ 18.3%→4.6%, |σ| 16.1%→3.6%. The 90°-rotated
  P-orientation sample stays cleanly distinct (real anisotropy, ~25% in H) —
  previously drift was a large fraction of that signal.
- **Window characterisation.** `thz.characterise_window(first_path, second_path,
  thickness_m=0.9e-3, theta_deg=45)` inverts the measured W for n_SiO₂(ω)−ik via
  `thz_core.invert_window_index` (iterative Fresnel-corrected, phase branch
  anchored by the measured envelope delay). CNT-13/D: n_SiO₂ = 1.962–1.967 flat
  over 0.5–2.7 THz ⇒ fused silica (crystal quartz would be ~2.1), so rotation
  sensitivity is geometric (wedge/mount), not birefringence. k comes out small
  but slightly negative at low f (−0.013→+0.01): the pure Fresnel model misses a
  ~1% coupling factor ⇒ **use the empirical W for reference construction; the
  model fit is for characterisation/diagnostics only.** n scales inversely with
  the assumed d (0.90±0.01 mm ⇒ ~1.1% systematic; measure d per window to do
  better). Measured inter-pulse delay 11.018 ps ⇒ n·d·cosθᵢ = 1.6516 mm.
- **Usage.** `transfer_function(dataset, config={"transfer":
  {"self_reference": True}}, ...)` on a `second_reflection` segment folder; the
  sibling `first_reflection` folder (same filenames — the `segment_reflections`
  layout) supplies the front pulses. Missing files raise (no silent fallback).
  The trusted mask is additionally intersected with the front-pulse SNR mask.
- **Sub-sample ramp interplay.** In self-reference mode the §4 spectral phase
  ramp is skipped: the front-pulse spectra are absolute-time referenced, so the
  correction already carries the true relative timing (applying the ramp too
  would double-count). Noticed in passing: in the two-phase workflow the stored
  residual never survives segmentation anyway (`processing_dict` is not
  persisted), so phase 2 has been running without the §4 ramp — self-referencing
  also repairs that hole naturally.
- **Gate hygiene.** Both gates must treat the GaP echo consistently (both
  exclude it, as the current gates do) so the multiplicative detector echo
  cancels in W. Windowed deconvolution is exact only when the Hann gate weighs
  the drift's time-domain echoes equally for both pulses — keep each pulse
  near the centre of its gate (synthetic test: off-centre pulses + harsh
  0.8 ps-correlation drift left a ~5% residual; centred + realistic drift <1%).
- Tests: `tests/test_window_selfref_workflow.py` (5), thz-core
  `tests/test_window.py` (9). Evaluation scripts in `explorations/`.

## §12  THzDataReflection container and root-directory loading

### Why it exists
The original pipeline pointed `DataSet` at a `second_reflection/` folder and
used the path-based `_first_reflection_spectrum` helper to reload the front
pulses from disk every time `transfer_function` was called with
`self_reference=True`. The first pulses were processed inconsistently: full-trace
`np.hanning` + mean subtraction in one go, with no baseline step, no global
truncate, and no user-configured window. The main (second-reflection) pipeline
applied `subtract_baseline → global_truncate → window_time (user config) →
zero_pad → fft_spectrum`. This mismatch meant the W = Y₂/Y₁ ratio was formed
from spectra treated differently.

### THzDataReflection (`dataset_core/data_structures/thz.py`)
`THzDataReflection(THzData)` subclass that stores the second reflection as the
primary data object (all inherited `THzData` operations and the `data` property
act on it unchanged) and attaches the first reflection as `first_segment: THzData`.

- `from_thzdata(second, first)` — classmethod that promotes two existing
  `THzData` objects without re-averaging. Copy is shallow (`__dict__.update`),
  so no wasted computation.
- `all_segments()` → `[('first_reflection', first_seg), ('second_reflection', self)]`
  — iterator protocol for pipeline functions that need to process both gates.
- Fully backwards-compatible: code that never checks `isinstance(...,
  THzDataReflection)` sees a normal `THzData`.

### Root-directory loading (`DataSet.load_all_data`)
`DataSet(root_dir).load_all_data()` now auto-detects whether
`root_dir/first_reflection/` and `root_dir/second_reflection/` subdirs exist.
If both are present, `_load_reflection_layout` pairs files by name and builds
`THzDataReflection` objects. Files in `second_reflection/` with no counterpart
in `first_reflection/` are loaded as plain `THzData` with a warning.

Pass `explicit_dir=True` to bypass detection and load `file_dir` as a flat
directory even when the subdirs are present — use this when re-running
`segment_reflections` preprocessing on raw files that live inside an already-
segmented directory tree, or when accessing raw acquisitions for debugging.

Legacy code that pointed directly at `second_reflection/` continues to work
unchanged: no `first_reflection/` subdir exists there, so the loader falls back
to the old flat-directory path, and `transfer_function` uses the file-based
fallback for front-pulse spectra.

### Pipeline consistency fix
`subtract_baseline` and `window_time` now iterate `THzDataReflection.all_segments()`
(via `isinstance` check) so both gates receive identical pre-processing.
`zero_pad` operates on the second reflection only (gates have different lengths
by design; padding is handled implicitly in the FFT step).

`fft_spectrum` calls `_compute_first_segment_fft(first_seg, freq)` after the
second-reflection FFT. This function back-calculates `n_fft = 2*(len(freq)−1)`
from the second's frequency grid and uses `np.fft.rfft(first_y, n=n_fft)` —
equivalent to zero-padding the shorter first gate to the same length — then
applies the absolute-time phase reference `exp(−i·2πf·t₀)`. The result is
stored in `first_seg.processing_dict['fft_spectrum']` on the same frequency grid
as the second reflection.

`transfer_function` with `self_reference=True` checks for the stored spectrum
first (`THzDataReflection` path), and only falls back to loading from disk if
it is absent (legacy path). The `_get_first_spectrum` helper encapsulates this
logic so the rest of the function does not change.

### n_window array fix (`_resolve_reflection_geometry`)
`theta_internal_rad` must be a scalar (Snell gives one angle). When `n_window`
is a per-frequency array, computing `float(np.real(snell(..., n_window)))` would
silently discard the array dimension. Fixed by extracting a scalar mean first for
the angle, while passing the full array to `fresnel_reflection_s` for
`r_reference`:
```python
n_window_scalar = float(np.real(np.mean(np.atleast_1d(n_window))))
theta_internal_rad = float(np.real(snell(..., n_window_scalar)))
r_reference_value = fresnel_reflection_s(n_window, 1.0, theta_internal_rad)
```
At the CNT-13/D window (n_SiO₂ = 1.962–1.967 flat), the array vs scalar change
shifts `r_reference` by +0.003 (+0.7 %) — comparable to the n measurement
uncertainty, so worth keeping but not urgent to backfill old results.

## §13  Single-pass in-memory reflection build + hardening (2026-06-16)

### Motivation — drop the save→reload round-trip
The two-phase workflow (segment to `.acc` files, then reload and process) had a
real cost: `processing_dict` is not persisted across the save, so the §4
sub-sample timing residual computed in phase 1 was silently lost before phase 2
(noted in §11). The new single-pass flow keeps everything in memory:
`load raw → define gates → build_reflection_dataset (crop in-memory) → align →
process both segments`. Each acquisition's individual scans are preserved because
`build_reflection_dataset` crops each scan's `raw_data` (per-scan, in ps) into the
first/second `THzData` segments up front — no averaging-away of scatter.

### §13a  `save_segmented` must run BEFORE the in-memory build  ★ bug fixed
We still want the **option** to cache the gated segments as `.acc` files (so a
later run can reload the segmented layout). The trap: `segment_reflections` gates
the trace it is given by **both** the first and second time windows. Once
`build_reflection_dataset` has run, each object's `raw_data` is *already* only the
second reflection (~159–168 ps), so re-gating it by the first-reflection window
(~151–158 ps) selects zero samples → `ValueError`. **Fix:** run the optional
`segment_reflections` save on the **raw flat dataset** (each trace still holds both
reflections) *before* `build_reflection_dataset`. Ordering is the whole fix; the
splitter itself is unchanged.

### §13b  Per-scan matrix staleness guard — one check, not N syncs  ★ latent bug
The per-scan working matrix (§ per_scan_working_matrix: `working_scans` =
`[time_s, scan1…scanN]`) is carried in lockstep with the averaged `data` by the
**linear** steps (baseline, alignment, normalise) so the segmented `.acc` files
retain every acquisition. But three steps change `data`'s **row count / time
sampling** and deliberately do **not** update the matrix:
`global_truncate` (rows ↓), `center_pulse` (prepend, rows ↑), and the
windowing-hygiene crop in `centering_manual` (index-crop, rows ↓).

We chose **not** to make those three steps also rewrite the matrix. Maintaining two
full parallel representations through every step is exactly the tight coupling /
"same data in two places" we avoid — each new step would have to remember to update
both, and a missed update is a silent corruption. Instead the invariant lives in
**one** place: `_ensure_scan_matrix` now verifies on every call that the matrix row
count matches `data`, and on a mismatch (or single-scan / absent `raw_data`) falls
back to the averaged trace as a single "scan" — losing the per-scan scatter but
never returning misaligned rows. The previous ad-hoc version of this check lived
inside `segment_reflections` only; centralising it protects **every** consumer
(present and future), which is the point. A one-shot note prints when the fallback
fires so the loss of per-scan power is visible, not silent.

In the current single-pass reflection flow this guard is mostly a safety net — the
only consumer (`segment_reflections`, when `save_segmented=True`) runs before the
row-changing steps — but it removes a live trap for the transmission flow and any
future code that touches `working_scans` after windowing/centring.
Tests: `tests/test_thz_reflection.py` (+3: fresh-matches-data, staleness-rebuild,
single-scan).

## §14  FFT T0 reference — audit of the asymmetric self-referencing  ★ (2026-06-16)

### The question
`fft_spectrum(segment='first_reflection')` multiplies the front-pulse spectrum by
the absolute-time factor `exp(-2πi·f·t[0])`; the second reflection gets **no** such
factor (its `rfft` is referenced to its own array sample 0). Does that asymmetry
leave a residual linear phase in `H = (Y₂ₛ/Y₁ₛ)/(Y₂ᵣ/Y₁ᵣ)` and bias n?

### How an FFT references phase (first-principles)
`rfft(y)` measures every spectral phase from the **first sample of the array** — the
segment's own "T0", which sits at absolute time `t[0]`. Two ways to choose that
origin: **own** (plain `rfft`, origin = array sample 0) and **absolute** (`×
exp(-2πi·f·t[0])`, origin = experiment time 0). They differ by a linear phase
`exp(-2πi·f·t[0])`. In the self-referenced double ratio the only surviving term is

```
H_abs / H_own = exp(-2πi·f·Δ),   Δ = (t₂ₛ[0]−t₁ₛ[0]) − (t₂ᵣ[0]−t₁ᵣ[0])
```

i.e. it depends **only** on whether the four segments start at different *absolute*
array times. If all four share a common origin, Δ=0 and the choice is irrelevant.

### Demo result (CNT-17 s-0, `explorations/demo_phase_referencing.py`)
Ran one sample + its reference through the real per-segment chain (baseline →
truncate → center_pulse → window), then formed `H` three ways (own / abs /
prod=first-abs-second-own) at two points:

- **PRE-pad** (FFT each segment at its own origin, right after windowing): the
  segments do **not** share an origin — `center_pulse` prepends a *different* number
  of zeros per file to centre each pulse (sample first-seg 49 samples, reference 51),
  so `t₁ₛ[0]−t₁ᵣ[0] = +0.10 ps`. Result: **Δ = −0.10 ps**, and own vs abs `H` differ
  by exactly a −0.10 ps linear phase (slope fit confirms). This is the feared bias.
- **POST-pad** (production order: `zero_pad` → `fft_spectrum`): **Δ = 0.0000 ps**,
  own ≡ abs ≡ prod (|H| diff 2e-15, phase slope 0.0000 ps). The methods collapse.

### Why production is already safe — and the real lever
`zero_pad` calls `pad_to_common_grid`, which places **every trace of a segment onto
one shared grid** by its absolute `t[0]` (earlier-starting traces get leading zeros).
So by the time `fft_spectrum` runs, all sample/reference traces of a given segment
share an identical array origin at the **same absolute time**, and the per-file
`center_pulse` prepend bookkeeping has been normalised away. The own-vs-absolute
choice (and hence the first-segment-only asymmetry) is therefore a **no-op in the
current pipeline**: the abs factor is a per-segment common constant that cancels in
the double ratio.

**Conclusion / decision (2026-06-16):** the asymmetric T0 reference is **not** a bug
in the current flow — `pad_to_common_grid` (always run before the FFT) neutralises
it. We do **not** need to make the reference symmetric for correctness. The genuine
invariant is *"common-grid the traces before the FFT"*; the FFT T0 factor is
redundant given that. Caveat to preserve: this safety depends on `zero_pad` running
**before** `fft_spectrum` and on `pad_to_common_grid` placing by absolute `t[0]`. Any
reorder that FFTs a segment **before** the common-grid step (or a future per-segment
FFT path) would reintroduce the −0.1-ps-class own-T0 error — at which point the
symmetric absolute reference becomes the right defence. `|H|` is phase-ramp
invariant throughout, so this is a pure n/group-delay concern, never `|r|`.

## §15  Front-reflection window extent & symmetry (limited pre-pulse)  ★ (2026-06-16)

### The constraint
CNT-17 (typical for us) gives only **~2.3 ps of pre-pulse** before the first
reflection (peak 154.3 ps, acquisition starts 152.0 ps) but **~11 ps** to the second
reflection. We can extend pre-pulse collection to ~8–9 ps before T0, but a **laser
pre-pulse** (a weaker THz replica) is a hard wall beyond that. So a *symmetric*
peak-centred window of maximum symmetric extent on the front pulse is only ~4.6 ps
wide and **discards ~9 ps of real post-pulse oscillation**.

### Demo (`explorations/demo_window_extent.py`, CNT-17 s-0)
Three front-pulse windows, same half-Hann taper logic, second reflection held at a
fixed symmetric window, then `H = (Y₂ₛ/Y₁ₛ)/(Y₂ᵣ/Y₁ᵣ)`:
- `narrow_sym` — symmetric ±2.3 ps (pre-pulse limited); drops the tail.
- `asym` — 2.3 ps pre + 8 ps post (keeps tail, asymmetric about the peak).
- `prepend_sym` — prepend zeros so pre = post = 8 ps, symmetric (keeps tail; the
  taper rise falls in the prepended-zero region, so the **real** pulse keeps
  near-full weight).

### Findings (counter to the naïve "symmetric is safest")
1. **`narrow_sym` is the worst**, not the safest: forcing symmetry by the short
   pre-pulse throws away real post-pulse signal → |Y₁| distorted **5.1 %** (vs
   `prepend_sym`), coarser resolution, low-f leakage. "Symmetric" is a false economy
   when symmetry is achieved by *discarding data*.
2. **`prepend_sym` is the most faithful.** Prepending zeros is **not** fabricating
   signal — the pre-pulse region is genuine baseline, and putting the window's rising
   taper there leaves the real pulse near full weight. `asym` keeps the same time
   span but its steep rise *tapers the real leading edge*, and its non-zero window
   mean **injects DC / low-f**. |Y₁| diff: `asym` 0.5 %, `prepend_sym` 0 (reference).
3. **Where it reaches the measurement.** Front-window choice barely touches the
   **trusted band**: phase(H) rms error 0.5–4.5 THz is **0.004 rad (asym) / 0.014 rad
   (narrow_sym)** — negligible (the front pulse is common to sample & reference, so
   self-referencing largely absorbs it there). The error is **concentrated below
   ~0.5 THz** (full-band rms ~0.3 rad) — i.e. it lands on the **low-frequency wing**,
   exactly the region we were worried about. `|H|`/n in the trusted band are safe.

### Design implication (leaning, pending confirmation)
Keep the **prepend+taper centering** — Samuel's earlier "orange flag" about it is
resolved: prepending into genuine baseline is safe and is in fact the *cleanest* way
to window a pulse whose acquisition truncates its pre-pulse. Lean toward making it
the **default for the first reflection when pre-pulse < target window** (it cleans
the low-f wing), with a guard/validation that the prepended region is really baseline
(no laser pre-pulse / echo) and a pre/post |Y₁| check. It is a low-f-wing refinement,
**not** a trusted-band n correction. The taper specifically earns its keep when the
acquisition truncates the *rising edge* (smooths the zero→pulse discontinuity).
The symmetry question that *does* bite the trusted band is the **second** reflection
(dispersive sample pulse vs bare SiO₂ reference pulse — different shapes, so an
asymmetric window shifts their centroids differently and does **not** cancel) — to be
workshopped next.

## §16  Windowing styles: cancellation vs mixing, per reflection  ★ (2026-06-16)

### Setup
`explorations/demo_windowing_styles.py` windows **both** reflections of the sample and
the reference with a chosen *style per reflection* (driven by `WINDOW_STYLES` /
`STYLE_COMBINATIONS` dicts; apodization via the real `thz_core.window_time`), keeps
both pulses on one shared axis, and builds `H = (Y₂ₛ/Y₁ₛ)/(Y₂ᵣ/Y₁ᵣ)`. Extents are
clamped to **±5 ps** — the clean isolation room, since the second reflection has only
~5.5 ps to the inter-pulse midpoint and the GaP echo sits ~5.3 ps after it (an 8 ps
window contaminated the second reflection with the first reflection's tail — a real
gate-hygiene trap worth remembering). Styles: `symmetric_prepended` (full, symmetric,
prepend zeros for the short front pre-pulse), `symmetric_shortest` (±2.3 ps narrow),
`asymmetric_tukey` (2.3 pre / 5 post, Tukey α=0.5).

### Result — trusted-band (0.5–4.5 THz) phase(H) error vs the ideal (prepend/prepend)
| combination | |H| rms frac | phase rms (full) | phase rms (trusted) |
|---|---|---|---|
| both narrow | 0.11 | 0.27 | 0.050 |
| both asymmetric | **0.48** | 0.37 | 0.063 |
| **narrow 1st, good 2nd** | 0.15 | 0.27 | **0.006** |
| **good 1st, narrow 2nd** | 0.05 | 0.07 | **0.052** |

### Findings
1. **Cancellation is asymmetric between the two reflections.** A consistent imperfect
   window on the **first** reflection cancels almost completely (trusted phase
   **0.006 rad**) — sample and reference front pulses are the same air→SiO₂ reflection,
   so identical distortion divides out. The **second** reflection does **not** cancel
   (**0.052 rad**): the dispersive CNT back-pulse and the bare SiO₂→air reference pulse
   have different shapes, so the same window distorts them differently.
2. **Mixing is safe iff the second reflection gets the faithful window.** `narrow 1st /
   good 2nd` is negligible (0.006 rad); `good 1st / narrow 2nd` carries the whole error
   (0.052 rad). So the rule is simply: **the second reflection sets the trusted-band n;
   the first reflection is forgiving.**
3. **Asymmetric windows wreck |H| magnitude** (0.48 frac rms) via DC injection /
   leakage, even though their trusted-band phase is only ~0.06 rad — so they are bad
   for k / |r| regardless. Prefer symmetric.
4. **All imperfections land mostly on the low-frequency wing** (<0.5 THz); trusted-band
   effects are modest (≤0.06 rad ≈ ~1 % n). Self-referencing is robust overall.

### Decisions (windowing, 2026-06-16)
- **Window the second reflection symmetrically and faithfully** — it is the one that
  reaches trusted-band n. It has ~5 ps of clean isolation (no prepend needed).
- **The first reflection is forgiving**; a narrow symmetric window is fine for phase/n.
  Prepend+taper there is a **low-f-wing** refinement, optional (§15), not critical.
- **Avoid asymmetric windows** (Tukey biased off-peak) — they inject DC and distort |H|.
- Size every window to the clean isolation extent (don't overrun the neighbouring pulse
  or the GaP echo); ±5 ps for CNT-17.

## §17  Shared-axis reflection isolation — implementation  ★ (2026-06-16)

Replaces the old `build (crop) → align → centering_manual → center_pulse →
window_time → zero_pad (pad_to_common_grid)` chain for reflection data with three
transparent functions (`dataset_core/adapters/thz_adapter.py`), kept **alongside**
the old path for A/B.

- **`build_full_trace_reflection(dataset, config=None)`** — wraps each raw trace into a
  `THzDataReflection` where BOTH segments are copies of ONE shared trace, isolated to the
  two pulses. `config['regions']` holds *generous, dataset-independent* search ranges; the
  function finds the non-zero data extent inside each (trimming zero-pad / region overhang),
  clips the shared axis to `[first_start, second_end]`, zeros the between-pulse gap (the
  inter-pulse delay is preserved as zeros, which self-referencing needs), and **self-modifies
  `config['regions']`** to the found (tightened) ranges — so the general search window never
  needs hand-tuning per dataset. Cross-file it uses the *intersection* of per-file found
  ranges (guaranteeing real data in every file's kept window). No echo crop is needed: the
  echo is excluded by the second-region trailing edge + the outside-region zeroing. If
  regions are absent/`None`, it falls back to the old whole-trace behaviour and leaves
  `first_region`/`second_region = None` for `define_reflection_regions` (interactive path).
- **`define_reflection_regions(dataset, config)`** — one function, both regions, from
  preset ps bounds or a `SpanSelector`. Writes the (start, stop) ps tuples onto every
  object as `first_region`/`second_region` and mirrors them into `config['regions']`.
- **`isolate_and_window(dataset, config, center_mode, show_graph)`** — the one
  coupling step. A **peak-anchored** window per reflection (value 1 *at the pulse
  peak*, tapering to zero at both window edges), both pulses placed on ONE shared
  axis so the inter-pulse phase is structural (no common-grid, no absolute-time
  factor, no sub-sample ramp). `center_mode='crop'` = symmetric window, half-width =
  the shorter of pre/post (narrower, no centroid shift); `'pad'` = the FULL region
  (keeps all data, taper rates differ per side when the region is asymmetric about
  the peak). **Region-zeroing**: amplitude outside `[region_start, region_end]` is
  zeroed before windowing — the region is the only echo/neighbour filter (no crop
  step). Every decision (peak, region, pre/post half, mode, window) is printed.

  **Taper fix (2026-06-16):** the first `'pad'` implementation grew the window to the
  *longer* half and padded the short side with zeros, then apodized with a symmetric
  Hann (`thz_core.window_time`). That left the Hann **non-zero at the region edge**
  where the data was cut → a sharp step on the padded side (and the symmetric Hann's
  crest sat off the pulse). Fix: the window edges sit on the data boundary and the
  taper is **peak-anchored** (split the Hann/Tukey cosine at the peak so each side
  tapers over its own width). Same cosine maths as `thz_core.window_time`, applied
  asymmetrically — both sides now taper smoothly to zero with the pulse at full
  weight. Consequence: `crop` and `pad` n now agree closely on CNT-17 (s-0 ≈ 0.95
  both), where the buggy pad had diverged.
- **`global_truncate` equal-length fix (2026-06-16)**: it now trims every trace to the
  common *minimum* sample count, not just by time value. A float boundary sample could
  land in some files but not others (off-by-one), leaving files on slightly
  different-length axes; once the dataset-specific 168 ps crop (which had masked this)
  was removed, `isolate_and_window`'s shared-axis assertion caught it. Trimming to the
  shortest common count guarantees one identical axis for every file.

### Why one call, both regions (not the `segment=`-twice pattern)
The shared axis must be common **across both reflections AND across all files** (so the
FFT grids match for `H = (Y₂/Y₁)…`). That coordination can only be done seeing
everything at once, so `isolate_and_window` is **two-pass**: (1) plan every window;
(2) build one common axis spanning all windows of all files, place every pulse on it.
The first attempt built the axis *per file* — but each file's peak is at a slightly
different time, so `pad` mode produced different-length axes per file and
`transfer_function` failed on a length mismatch. The shared two-pass axis fixes it.
(`crop` mode never extends the axis, so it worked even before the fix.)

### Checkpoint (`show_graph=True`)
Two figures, each one subplot row per file (sample and reference on their own rows):
(1) centred & tapered pulses on the shared **time** axis; (2) the pulses with their
window functions on a sample-**index** axis (the existing `window_time` style). One
call has all files in hand, so sample+reference are subplots in one figure — we do
*not* call per region.

### Smoke (CNT-17, `explorations/smoke_isolate_and_window.py`)
Full path build→regions→isolate→FFT→self-ref transfer→invert runs in both modes;
both reflections verified on one identical axis; `crop` gives ~40 trusted bins,
`pad` ~50 (wider window → finer resolution). Unit tests still 17/17.

### Wired into `run_me_reflection.py` (A/B), 2026-06-16
`pipeline_config['processing_path']` selects `process_shared_axis` (new) vs
`process_segmented` (old); both share `report()` and the same geometry/window config.
Shared-axis FFT zero-pads inside `fft_spectrum(n_fft=...)` (no `zero_pad`/
`pad_to_common_grid` step). Headless A/B on CNT-17 (0.5–3 THz band-mean n):
s-0 0.917 vs 0.964; s-180 0.696 vs 0.807; s-45 0.786 vs 0.840; s-90 0.763 vs 0.804 —
the two paths **agree to ~0.04–0.11 in n and ~0.05 in k**, with the same anisotropy
ordering, so the new shared-axis path is validated against the existing one.

### `pad_trace_start` — extend the trace start backwards  (2026-06-17)
Optional step after `subtract_baseline` (toggle `enabled=`): ramp the leading
`taper_ps` (1 ps default) of real data up from zero, then prepend `extension_ps`
(3 ps default) of zeros. Creates a baseline-zero pre-pulse region so a later
*symmetric* window can reach back without cutting real data (mainly for the first
reflection's short pre-pulse). Real samples keep their absolute times; only the axis
extends earlier. Uses the `segment=`-twice pattern (transparency > one call over both
segments — Samuel's call); in the shared-axis path call it for both segments with the
same args so they stay on one axis. The region itself isn't moved, so to *use* the new
room you widen the first region into it.

**Step-change diagnostic (metric decision).** `show_graph` draws the padded traces
(zeros + taper shaded) and, below, the **first difference normalised to the peak**:
`(y[i]−y[i−1]) / max|y|` — reads directly as "the step as a fraction of pulse height",
with a dashed reference at the typical steepest *in-pulse* slope (anything in the
junction poking above it is sharper than any real feature). Samuel's first idea
(normalise to the *previous* step, `Δ[i]/Δ[i−1]`) was rejected: the prepended region
is zeros, so `Δ[i−1]≈0` there and the ratio blows up across the whole flat region,
hiding the one junction you want to see. CNT-17 junction steps come out 0.2–0.5 % of
peak (clean taper). One-shot numeric printed per file.

### `center_pulse` — centre in BOTH directions  ★ bug fixed (2026-06-16)
`_center_pulse_trace` only ever **prepended** zeros: `n_prepend = max(0, n_after −
n_before)`. That centres a pulse only when its peak is in the **first half** of the
array. When the peak is **past the midpoint** it clamped to 0 and printed "pulse
already centred, skipping" — misleading: it could not centre, it just gave up. This
bit the **segmented path's second reflection**: its gate (e.g. 161–168 ps) puts the
peak at ~63 %, so it was left off-centre, and `window_time` (whole-trace gate →
Hann centred on the array midpoint) then apodised it **asymmetrically** — exactly the
§16 centroid-bias failure, and *differently* from the first reflection (peak ~35 %,
which did centre). Fix: pad the **shorter** side — prepend if the peak is early,
**append** zeros if it is past the midpoint — with the half-cosine taper at whichever
junction is padded (rising at the front, falling at the back, clamped so it never
crosses the peak). Verified on CNT-17: second reflections now *append* ~31–35 samples
and centre; first reflections still prepend. Tests 17/17 + synthetic prepend/append/
already-centred cases. (This only affects the **segmented** path; the shared-axis path
does not use `center_pulse`.)


## §18  Transmission phase handling — group-delay preservation + intercept removal  ★ (2026-06-17)

Working a **free-standing transmission** measurement (fused-silica slab, d = 2.08 mm)
through the general pipeline. Two symptoms: (1) `n` came out near the expected ~1.9
but **drooped toward lower values approaching DC**, where theory says it should be
flat; (2) the existing interactive `phase_correction` did not fix it.

### How the group delay survives `centering_manual` (it is NOT lost)
`centering_manual` (formerly `pre_window_align_peak`) is **window hygiene only**: it
aligns every trace's pulse peak to a common **array index** and crops to equal
start→peak→end sample counts, so the window lands identically on every file. It does
**not** rewrite the time-axis values — but because it crops a *different* number of
leading samples per file, each file ends up **starting at a different absolute
`t[0]`**. After centering, the sample↔reference group delay lives entirely in that
`t[0]` difference.

A plain `np.fft.rfft` references phase to **array index 0** and **ignores the absolute
time column**, so FFT-ing at this point would drop the group delay and collapse `n`→1.
What saves it is `zero_pad` → `pad_to_common_grid`: it lays every trace back onto **one
shared absolute-time axis** (union span) by inserting each at offset
`(t[0] − t_lo)/dt`, turning the start-time differences into **index offsets**. The FFT
then sees the true `exp(−iωΔt)` ramp, and `n ≈ 1.9` comes out correct. This is the
**structural equivalent of the legacy `phioffset = 2πf·(t0_sam − t0_ref)`** (Ballabio
`main_TDS.py`), which adds the same term explicitly from the absolute time column.

**Refactor (2026-06-17):** split the adapter `zero_pad` into two named steps so this
physics-critical step is explicit and separately callable:
- `align_to_common_time_axis` — Part 1, wraps `pad_to_common_grid` (the
  group-delay-preserving shared-axis lay-down). Must run before `fft_spectrum`.
- `zero_pad` — Part 2, the optional trailing-zero resolution padding; it now calls
  Part 1 first, so all existing call sites stay backward-compatible.

### The low-frequency `n` droop = a constant phase offset on H
With `φ_H(f) = −2πf·Δt + φ₀`, the inversion `n = 1 − c·φ_H/(2πf·d)` picks up a term
`−c·φ₀/(2πf·d)` that **diverges as 1/f** toward DC. A non-zero intercept `φ₀` (from the
sub-sample timing residual, a wrap-branch error, or a slightly-negative baselined mean
biasing the DC bin) therefore droops/raises `n` only near DC while leaving it ~flat
elsewhere — exactly the observed symptom. Physically `φ_H` must pass through the origin
(a passive sample imposes no phase shift at zero frequency).

### Fix: `remove_phase_offset` (the legacy `phaseex`, done cleanly)
Legacy `phase_interpolation.phaseex` fits the sample−reference phase over 0.3–2 THz and
subtracts **only `lsq.intercept`** (keeps the slope = group delay). Re-implemented in
`thz_core.transfer.remove_phase_offset(f, H, fit_band_hz=(0.3e12, 2.0e12), mask=None)`:
unwrap H's phase, `np.polyfit` a line over the trusted band (∩ optional SNR mask),
subtract the intercept, rebuild `|H|·exp(i(φ−intercept))`. Magnitude untouched; NaN bins
passed through; no-op (`applied=False`) if <2 fit bins. thz-core tests 180 pass (+5).

Adapter `thz.remove_phase_offset` operates on `processing_dict['transfer_H']` — the
array `invert_nk` actually reads — run **after** `transfer_function`. Config:
`config['phase_offset']['band_thz']` (default (0.3, 2.0)) and `['use_snr_mask']`.

### Why the old `phase_correction` "did nothing"
`phase_correction(source='fft')` edits each trace's stored `fft_spectrum` **after**
`transfer_function` has already built `transfer_H`; `invert_nk` reads `transfer_H`, so
the edit never reaches the inversion. `source='transfer'` does hit `transfer_H`, but the
function is interactive (drag a span) and per-file, so the fit band is inconsistent. The
interactive `phase_correction` stays as a manual exploration tool; `remove_phase_offset`
is the deterministic, pipeline-friendly replacement that matches `phaseex`.

### ★ The droop was a whole-CYCLE (2π) wrap error — and it can't ride through complex H
Diagnosing the fused-silica slab (d = 2.08 mm) on real data exposed the subtlety that
makes this geometry-independent and important. The fitted intercept was **6.357 rad ≈
2π**, not a small number: the true phase at 0.3 THz is −12.3 rad (n≈1.94), but the slab
is thick enough that the phase exceeds π *before the first reliable bin*, so `np.unwrap`/
`robust_unwrap` settle on a branch **one full cycle too high**. Subtracting that intercept
from the *real unwrapped phase* gives **n = 1.94 dead flat** — correct.

But `remove_phase_offset` applies the correction by multiplying complex `H` by
`exp(−i·6.357)`, and **a 2π multiple is invisible in a complex number**
(`exp(−i·2π)=1`). `invert_nk` then takes its *own* `np.angle(H)` and re-unwraps, re-anchors
at the first bin, and lands right back on the wrong cycle → n unchanged (1.575→1.868,
still drooping). **The integer-cycle part of a phase correction can never travel through
a complex H array.** It must be fixed where the real unwrapped phase is decided.

**Fix (thz-core `invert_nk`, default on):** `config['invert']['anchor_phase_origin']`
(default True) fits a line to the trusted unwrapped phase, extrapolates to DC, and
subtracts the **nearest integer number of 2π cycles** so the phase passes through the
origin (a passive sample imposes no phase shift at f=0). Slope/group delay untouched;
thin samples already on the right branch (|intercept|<π → round to 0) are unchanged.
Measured effect on the slab: anchor OFF n=1.575→1.868 (droops); anchor ON n≈1.94 flat;
anchor ON + `remove_phase_offset` n=1.941 flat (the sub-cycle residual cleaned up too).

**Division of labour:** `invert_nk` origin anchor = the integer-cycle (thick-sample)
fix, the one that removes the droop; `remove_phase_offset` = the sub-2π fractional
residual. They are complementary. Verified: thz-core 180 tests pass (incl. pre-existing
`test_thin_sample_unaffected_by_anchoring`, `test_lossy_thick_sample_recovered`); the
snr-weights unwrap test now disables the anchor in its control assertion so it still
isolates the `robust_unwrap` weighting mechanism (the anchor is a second, independent
safeguard against the same branch flip).

### Are we over-engineering vs legacy? (investigation, `explorations/explore_phase_unwrap_vs_legacy.py`)
Samuel asked why the legacy `main_TDS.py` produced flat n with no anchor — are we
over-doing it? The exploration compared, on the SAME stored spectra, four phase
methods. Findings (CNT/silica slab, 6.55 ps inter-pulse delay):
- **Tested and REJECTED hypothesis:** the legacy's robustness is NOT from its
  reduced-phase unwrap (subtract `2π·f·t_peak`, unwrap residual, add back). That
  method gives the **same** φ(H) DC intercept (+6.357 rad ≈ +1.01 cycle) and the
  **same drooping n** (1.575→1.868) as our raw `np.unwrap`. The 2π error is not an
  unwrap failure — the phase genuinely starts beyond π at the first usable bin, and
  no unwrap scheme can recover the absolute cycle without asserting φ(0)=0.
- **What actually flattens legacy n:** `phaseex` removing the full ~2π intercept
  from the **real** phase array, in one place, immediately before `n = 1 − cφ/(ωd)`,
  **never round-tripping through a complex number.**
- **Why ours takes two steps:** our pipeline stores H complex and `invert_nk`
  re-unwraps, which splits the one phaseex correction into (a) the integer-cycle
  half — invisible in complex H, so handled by the `invert_nk` anchor — and (b) the
  sub-2π half — survives complex H, so handled by `remove_phase_offset`. The anchor
  alone already flattens n (~1.94); `remove_phase_offset` only trims the ~0.07 rad
  residual.

**Conclusion:** not physically over-engineered (same result as legacy), but the
complex-H round-trip + invert_nk re-unwrap is genuinely more moving parts than
legacy's single real-phase `phaseex`→n. A future simplification would be to remove
the intercept once on the real unwrapped phase and carry it to `invert_nk` without
re-unwrapping — deferred; current path is correct and tested. In practice the
default-on anchor alone is sufficient; `remove_phase_offset` is optional polish.


## §19  Substrate-sandwich model: opt-in Fabry-Pérot, explicit medium, continuity tracking  ★ (2026-06-19)

Extended the already-wired substrate-sandwich grid inversion (`multilayer.py` +
`invert_grid.py`; see `docs/sandwich_extraction_explained.md`) with three additions,
all **additive and backward-compatible** — the defaults reproduce the previous
output (regression-guarded, see below), and a config key turns each on for a
deliberate run. No adapter change: `thz_adapter._grid_invert_*` already forward
`config` verbatim, and the sandwich step spreads `**config.get('invert_grid', {})`,
so the new keys flow through untouched. (Single source of defaults = the core; the
config carries only deliberate overrides.)

### 1. Fabry-Pérot is OPT-IN (`fabry_perot: False` default) — and why that flipped
Initial instinct (2026-06-18) was FP **on** by default. Reversed it: we often run
**thicker** samples whose internal echo is well separated in time and is **gated
out** — that is the no-FP (gated) limit, and it must stay the default so existing
pipelines are unchanged. The thin-sample case (echo overlaps the main pulse, can't
gate) is the exception, so it opts in. FP re-enters via the existing
`fabry_perot_factor` primitive.
- **Which layers get an etalon term:** only the **thin** layer(s) being
  characterised. The substrate slabs are always the thick, gatable element (mm-scale,
  echoes separate), so they carry **no** etalon term even with `fabry_perot=True`.
  - ASMSA (sample step): etalon on the sample layer **and** the empty reference gap.
  - ASASA (substrate step): etalon on the **gap** only (matches MATLAB
    `Cuvette_code`'s single `(1 ± r·r·P²)` term); substrate echoes assumed gated.
  - free_standing: optional slab etalon (thin free films).
- **Sign-convention caveat (flagged for step-2 validation):** our
  `fabry_perot_factor` gives the textbook etalon `1/(1 − r²P²)` (resonant poles).
  MATLAB `Cuvette_code` writes the gap denominator as `(1 − r23·r34·P²)` which, with
  `r34 = −r23`, is `(1 + r23²P²)` — the **opposite** sign, no poles. This is a
  genuine convention discrepancy, NOT yet resolved by derivation. Deliberately left
  to **numerical validation on the Vasilis 200 K triplet** (step 2): if FP-on n(ω)
  disagrees with MATLAB, the sign is the first suspect. Did not silently "correct"
  either side.

### 2. Explicit surrounding medium (`medium_index: 1.0` default = vacuum)
Replaced the hard-coded air index `1` in the ASASA surround and the ASMSA empty-gap
reference with one `medium_index` parameter (covers both — if you're in vacuum,
everything outside the solid is vacuum; no physical case splits them). Default `1.0`
= vacuum; pass ≈1.00027 for dry air at THz. Rationale: the medium sits in the empty
**reference** gap and so does **not** fully cancel in the filled/empty ratio — it is
a small but **systematic** phase bias `(ω/c)(n_med−1)·d_gap` on the extracted index
(~0.017 rad for a 1 mm gap at 3 THz). Making it explicit documents the assumption
instead of burying `1` in the math. "Air" measurements are treated as vacuum for now
(dispersive air deferred). The ASASA closed form was re-expressed from primitives
(`t(n_med,n_sub)·t(n_sub,n_med)`)² · exp(−i(n_sub−n_med)·2d·ω/c); with `medium=1` it
is bit-identical to the legacy `[4n/(n+1)²]²` form (regression test asserts rtol 1e-12).

### 3. Branch-continuity tracking (`continuity_tracking.enabled: False` default)
New `_select_by_continuity`: instead of the global residual minimum at each bin
(which can hop between branches of the transcendental equation frequency-to-frequency
when FP is kept), restrict the candidate search to a `±search_half_width_cells`
window around the **previous bin's** solution. The first trusted bin is seeded by the
analytic guess (`_analytical_initial_guess`) so tracking starts on the physical
branch; the loop walks `trusted_indices` ascending (frequency upward), which is the
order continuity relies on. Empty-window fallback → global minimum (a bin is never
left unsolved). Default off = previous global-min behaviour preserved.

### Tests (thz_core: 196 → 196 pass; +16 new across multilayer + invert_grid)
- Regression guards: default medium reproduces the legacy ASASA closed form and the
  ASMSA air-gap reference **exactly** (rtol 1e-12).
- medium: air vs vacuum differ; H→1 when n_sub=n_medium.
- FP: requires `thickness_gap_m` when on (ASASA); changes the result; unity at no
  contrast; explicit factor-form check (ASMSA); substrate-only and sandwich **round
  trips** recover the true index with FP on.
- continuity: unit tests on `_select_by_continuity` (local-over-global pick;
  empty-window fallback); sandwich recovery with tracking enabled; bad half-width raises.

### Step 2 — validation against MATLAB on the 200 K triplet  ★ DONE (2026-06-19)
Script: `explorations/validate_sandwich_against_matlab.py`. It (a) replicates the
MATLAB preprocessing exactly (pad → peak-centred Hamming → 2¹² FFT → 0.8–2.0 THz crop)
so both solvers see the same `H = E_trans/E_ref`; (b) runs a faithful Python **port**
of each MATLAB grid solver (their exact Fresnel/FP expressions + ±N continuity) as the
reference; (c) runs `thz_core.invert_nk_grid` through the same `H`; (d) compares.
Findings (run on `Vasilis_Data/data`, 200 K):

- **Gated-limit model equivalence (pure algebra, no data):** `max|H_matlab −
  H_thzcore|` over the candidate grid = **7.8e-6**, and that residual is entirely the
  **speed-of-light constant difference** (MATLAB `2.997925e8` vs thz_core
  `299792458.0`, ~1.4e-7 relative, acting on the ~150 rad substrate phase). The no-FP
  models are otherwise identical. → thz_core asasa **is** the MATLAB substrate model in
  the gated limit.
- **Substrate, gated limit:** thz_core vs MATLAB-port `|Δn|` median **1.0e-4** (= grid
  resolution), `|Δk|` median 5.6e-3. Agreement to grid resolution in the bulk.
- **FP sign (substrate):** thz_core (textbook `1/(1−r²P²)`) vs MATLAB-as-shipped
  (`1/(1+r0²P²)`) `|Δn|` median **1.3e-3** — small but systematic, as predicted. The
  FP correction itself (thz_core on vs off) is `|Δn|` median 4e-4. **Verdict: the
  substrate MATLAB has a FP-sign slip; thz_core's consistent textbook sign (which also
  matches the *sample* MATLAB code) is correct.** We feed the **gated** (FP-off)
  substrate index to the sample step anyway — the 1 mm slabs have well-separated,
  gatable echoes, so FP on the substrate is not the right model regardless.
- **Sample (ASMSA), the actual measurement goal:** thz_core FP-on vs MATLAB-port FP-on
  `|Δn|` median **1.4e-3** (≈ grid resolution), `|Δk|` median 4.7e-4 — they overlay.
  FP-off vs MATLAB(on) is 10× worse (`|Δn|` median 1.4e-2), confirming FP matters at
  the ~1e-2 level here and thz_core-on tracks it correctly. Sample `n ≈ 2.16`, `k`
  rises to the 0.2 search ceiling.
- **Substrate sawtooth — diagnosed (2026-06-19), NOT the search range, NOT a thz_core
  bug.** Samuel's measured params: substrate n ≈ 1.95, slab **0.9 mm** (not 1 mm), gap
  60–120 µm (gap irrelevant to substrate n in the gated limit — it cancels). Re-centring
  the range on 1.95 with d=0.9 mm moved the median to ~2.0 but the **sawtooth persisted**.
  Root cause, from `n` via direct phase slope: the substrate `H` from the MATLAB's
  manual **asymmetric zero-padding** (`150`/`210` samples) is a crude time-alignment that
  does NOT match the true ~5.7 ps slab delay, so the unwrapped phase is cycle-ambiguous
  (pinned ~0 at 0.8 THz when it should be ~−41 rad — off ~13 cycles) with a ~10 rad
  non-linear residual. The grid seed at bin 0 therefore rails at the upper bound, and
  continuity then **slides at exactly its window speed-limit** (max per-bin Δn = ±0.008,
  0 violations — continuity is provably correct). So the input `H`, not the inverter, is
  the problem. This is preprocessing-limited (the same class of issue as
  [[transmission_phase_chain]] / [[robust_unwrap_design]]).
- **Sample n is INSENSITIVE to the substrate sawtooth (the key result).** Feeding a flat
  `n_sub = 1.95−0.01i` vs the sawtoothing `n_sub` gives the **same** sample `n`
  (median 2.158 both; `|Δn|` med 0.002 = grid resolution, `|Δk|` med 0.0). The
  differential filled/empty ratio cancels the thick-substrate phase, so the substrate
  weak point does NOT compromise the science. → the sandwich pipeline is validated and
  fit for sample extraction as-is.

Overlay figure: `explorations/validate_sandwich_against_matlab.png`.

### Step 2b — legacy Novelli cross-check  ★ (2026-06-21)
Script: `explorations/run_legacy_novelli_two_step.py`. Drives the group's legacy-thz
helpers (M. Ballabio; `dataimport`/`fft_err`/`padding`/`phase_interpolation`) in a
two-step workflow WITHOUT editing `main_TDS.py`. Step 1: air ref + substrate sample,
ns=1 → free-standing slab (the Novelli Fresnel factor collapses to `(n+1)²/4n`). Step 2:
substrate ref + sample sample, ns=n_sub(ω). Analytic, no FP (its honest nature; an
independent baseline). Findings:

- **The Novelli FORMULA is correct.** Fed a properly preprocessed sample H (MATLAB-style
  symmetric padding), the analytic `n = 1 + Δφ·c/(ωd)` returns **n_sample ≈ 2.15**,
  matching thz_core/MATLAB (2.16). The physics agrees across all three methods.
- **The legacy bundled PREPROCESSING is the weak link, not the inversion.** Run end-to-end
  with the legacy's own `centerpad` + `phaseoffset`, the sample comes out
  **n ≈ 6.71 (non-physical)** — the centre-pad/offset mishandles the inter-pulse delay on
  this thick-substrate cuvette data. Decisive test: same formula, MATLAB preprocessing →
  2.15; legacy preprocessing → 6.71. So legacy-thz as-shipped is NOT reliable here; its
  time-domain front-end is the problem.
- **Substrate measurement is TWO slabs** (empty cuvette = air|sub|gap|sub|air): step 1
  must use the 2-slab path (2·0.9 mm). With that, legacy gives a smooth flat
  **n_sub ≈ 2.47**; the legacy phase is clean and linear (slope = absolute, intercept ≈ 0,
  no cycle ambiguity).
- **This explains the grid sawtooth: the true n_sub is ABOVE the grid ceiling.** Extending
  the thz_core substrate range past 2.02 (to 2.6 / 3.0, identical results) un-rails it →
  n_sub ≈ **2.21** (MATLAB preprocessing). So n_sub is preprocessing-sensitive
  (legacy 2.47 vs grid 2.21) but unambiguously **≫ the 1.71–1.80 the MATLAB searched and
  > the 1.95 estimate** — the original sawtooth was the 2.02 ceiling clipping the answer.

### Step 2c — substrate puzzle RESOLVED: it is fused silica; the inflation is preprocessing  ★ (2026-06-21)
Samuel: the substrate is fused silica (SiO₂), known n ≈ 1.96 at THz (cf.
[[window_selfref_evaluation]] n_SiO2=1.964) — so the 2.2–2.5 from grid+legacy is wrong.
Decisive check = **raw time-domain peak-to-peak delays** (preprocessing-independent):
- substrate↔air peak delay = **5.3 ps** → over the 1.8 mm two-slab path, **n_sub = 1.88**
  (and n=1.96 needs 1.66 mm = 0.83 mm/slab, consistent with ~0.9 mm windows). = fused silica. ✓
- sample↔substrate peak delay = **0.5 ps** → **n_sample = 2.15** ✓ (matches all methods).

So **the inflated substrate n (grid 2.21 / legacy 2.47) is a PREPROCESSING artifact**: both
the MATLAB asymmetric zero-pad (hardcodes ~3 ps shift) and the legacy `centerpad` push the
substrate phase ~3 ps beyond the true 5.3 ps delay. Extending the grid range only let it
settle on the inflated value instead of railing — it did NOT fix the cause. The genuine
substrate n is ~1.9 (fused silica). Sample n is unaffected (differential cancels it).

### Step 2d — thz_core's OWN preprocessing gives the correct n  ★ CONFIRMED (2026-06-21)
Script: `explorations/extract_cuvette_via_thzcore_preprocessing.py`. Ran the same 200 K
triplet through thz_core's real transmission front-end — each pulse windowed IN PLACE on the
shared time axis (`window_time`), `fft_spectrum`, `transfer_function`, analytic `invert_nk`
— with NO asymmetric padding. Result:
- **substrate n = 1.949, flat [1.946, 1.959]** over the 2-window 1.8 mm path = **fused silica ✓**
  (the 0.9 mm 1-slab case gives 2.90 ≈ 2×, confirming the two-window path).
- **sample n = 2.151** ✓ (matches all methods).

So thz_core's careful in-place windowing introduces no spurious group delay and recovers the
textbook value, where the MATLAB/legacy hardcoded padding does not. **Confirmed: the
front-end was the entire discrepancy** (Samuel's instinct — both pipelines give ~1.96 on a
normal silica slab — was right; this cuvette data just exposed the crude-padding front-ends).
Note Samuel previously cross-checked both pipelines on another fused silica at ~1.96.

### Next (paused for assessment)
Sample extraction is validated and robust across all three methods (thz_core ≈ MATLAB ≈
Novelli-formula ≈ 2.16). Substrate is fused silica (~1.9); grid/legacy over-estimates are
preprocessing artifacts (raw peak delay confirms). Open items for Samuel: (1) keep the textbook FP sign
(recommended). (2) Substrate n is genuinely uncertain (~2.2–2.5, preprocessing-sensitive)
and higher than assumed — **extend the substrate search range to ~2.6** and ideally
measure the substrate independently; what material/expected n is it? (3) The legacy code's
value as an independent check is limited to its (correct) formula — its preprocessing
needs replacing to be usable on cuvette data. (4) Sample step can use a smooth/constant
n_sub with no measurable effect. Drude-Smith / conductivity (downstream) out of scope.

---

## Presentation: true resolution + frequency-domain error (2026-07-03)

**Resolution vs bin spacing (the correction).** A large `n_fft` makes the FFT *bin spacing*
`df_fft = 1/(n_fft·dt)` far finer than the physical *resolution*. Zero-padding is sinc
interpolation — no new information. The true resolution element is `df_res ≈ 1/T_window`.
For `H = W_sample/W_reference` (each `W = Y₂/Y₁`) the resolution is set by the SHORTER of
the two reflection windows; the inter-pulse gap cancels in the ratio and buys no resolution.
`compute_instrument_resolution` records `T_res = min(first_region_len, second_region_len)`,
`df_res`, and `k = round(df_res/df_fft)`; plots mark one point per `k` bins, and the opt-in
`apply_instrument_resolution` decimates the stored arrays. CNT-21: `df_res ≈ 0.155 THz`,
`k = 16`. (Hann apodization broadens the element to ~2·df_res — set `broadening_factor=2`
for the conservative width.)

**Faithful frequency-domain error (answering "how do we propagate error into frequency,
given on-peak noise ≫ the floor?").** The two domains behave differently, and that resolves
the dilemma: white additive TIME-domain noise maps (Parseval) to a FLAT spectral floor, so
the off-peak spectral floor *is* the additive measurement noise imaged into frequency. In
frequency there is therefore ONE unambiguous σ per spectrum (no time-domain "max/σ_at_max
vs max/floor" choice). `trusted_band_mask` already measures it as amplitude SNR
`snr_db = 20·log10(|Y|/floor)`, giving relative error `1/SNR_lin` per spectrum.
`compute_transfer_uncertainty` combines sample and reference in quadrature:
`(σ_|H|/|H|)² = 10^(−samp_snr_db/10) + 10^(−ref_snr_db/10)`, `σ_phase = √(that)` rad. The
bars grow exactly where a spectrum is weak — self-consistent with the SNR mask.

*Limitation (Samuel's on-peak worry is legitimate):* the floor captures only ADDITIVE noise.
On-peak elevated scatter (timing jitter, amplitude drift) is signal-correlated and NOT in
the floor. The faithful upgrade is to propagate the measured per-timepoint repeat scatter
`σ_t(t)` (higher at the peak) through the windowed DFT: `σ_Y(f)² ≈ ½·Σ_t w(t)²·σ_t(t)²`.
That needs `stderr(t)` plumbed through baseline + window (currently dropped) — documented
TODO. n/k/ε/σ error propagation is likewise a follow-up (only `|H|`/phase have bars now).

**Plot style.** One helper `_plot_with_snr_mask` feeds every frequency plot: faint guide
line (unbiased) + solid markers on the resolution grid (they carry the judgement) + hollow
faded markers and a light grey `axvspan` over SNR-excluded bands (visible, not hidden) +
optional error bars. Choices were Samuel's: decimate (not bin-average), faded markers +
shaded band, noise-floor error bars.

---

## Kramers-Kronig phase correction (analytical fitting method) (2026-07-03)

Packaged the misplacement phase correction of Jatkar, Yeh, Pancaldi & Bonetti (*Robust
phase correction techniques for THz-TDS in reflection*, arXiv:2412.18662) into the
pipeline as `phase_correct_kk` (opt-in, between `transfer_function` and
`invert_nk_reflection`).

**Method.** In reflection geometry an unknown sample-vs-reference shift `l` adds a purely
linear phase `phi_m = phi_i + (omega/c)(2l/cos theta)` (paper Eq. 2). The KK relation on
`ln r = ln|r| - i*phi` ties the correct phase to the reliably measured `|r|`. Define
`Delta_m(w) = (2/pi)PV∫_0^wend Omega*phi_m/(Omega^2 - w^2) dOmega - ln|r_m|`; the
misplacement makes it `≈ (2l/(pi c cos theta))·w·ln((wend-w)/(wend+w)) + const` (Eq. 16),
so a LINEAR fit of `Delta_m` vs that analytical basis recovers `l`. Corrected phase
`phi_i = phi_m - (omega/c)(2l/cos theta)`.

**Polarization: independent.** The correction acts on `|r|` and `phi` only; the
misplacement term has no polarization dependence and the applied phase ramp is even
independent of `theta` (theta only rescales the reported `l`). Polarization enters ONLY the
downstream `r->n` inversion — s-pol Eq. 20a vs p-pol Eq. 20b — which `invert_nk_reflection`
already implements (`polarization='s'|'p'`). The paper validates s AND p at 45°. Verified
end-to-end on real p-pol CNT data through the p-pol inversion.
*Caveat:* KK assumes minimum phase (no zeros of `r` in the upper half-plane). Strained for
p-pol near the (pseudo-)Brewster dip (rapid ~pi swing) and for `r<0` (angle≈pi constant
offset = the paper's `phi_0` term, which the caller must handle). Fine for `|r|≈1`
rotating-phase samples (metals, conductive CNT).

**The core (`thz_core.kramers_kronig`) already existed but had TWO bugs — both fixed:**
1. `estimate_misplacement` fit `Delta_m` with a raw `lstsq` on `[basis, ones]`, where the
   basis column is `omega*ln(...) ~ 1e13` vs a ones column of 1 — catastrophically
   ill-conditioned; `rcond=None` dropped the intercept's singular value and returned a
   SIGN-FLIPPED, ~10-25%-wrong slope. Fixed by `np.polyfit` (scales the columns).
2. `correct_reflection_phase` ADDED the misplacement phase instead of subtracting it,
   DOUBLING the error rather than removing it. Fixed the sign.
   The old `thz_core/tests/test_kramers_kronig.py` had ENCODED both bugs as "known
   limitations" (`assert l_est < 0`; "recovers to within 10-25%"); rewritten to assert the
   correct behaviour (difference method cancels the finite-band baseline bias).

**Regimes / limits.** Accuracy is set by BANDWIDTH (`f_end` must sit above spectral
features), not window length. The ABSOLUTE recovered `l` carries a finite-band bias (paper
Eq. 13, the intrinsic-phase tail above `f_end`); the DIFFERENCE of two estimates cancels it.
Large misplacements (tens of um → hundreds of fs) wrap the phase many times and degrade the
unwrap. On real p-pol CNT it recovers a consistent ~40 um (~390 fs) offset across repeats.
Tests: `tests/test_phase_correct_kk.py` (4/4) + rewritten `thz_core` KK tests (3/3).

---

## Phase-only self-referencing (`self_phase`) + the align_to_reference bug (2026-07-03)

**New `config['transfer']['self_phase']` mode.** Between plain and full self-reference:

    self_reference : H = (Y2_s/Y1_s)/(Y2_r/Y1_r)      front pulse for TIMING + amplitude
    self_phase     : H = (Y2_s/Y2_r) · (C/|C|), C=Y1_r/Y1_s   front pulse for TIMING only
    plain          : H = Y2_s/Y2_r                    (needs align_to_reference for timing)

Rationale: on the shared axis a rigid acquisition timing offset Δt multiplies BOTH front
and back sample spectra by `exp(-2πifΔt)`. The plain back ratio carries `exp(-2πifΔt)`
(spurious linear phase → wrong n); the unit-magnitude front correction `C/|C|` carries
`exp(+2πifΔt)`, so the product CANCELS the offset — WITHOUT importing the front-pulse
amplitude `|C|` that full self-referencing folds in. So the first pulse is used purely as a
timing (phase) reference; normalisation stays on the back reference. It is a structural,
frequency-domain replacement for time-domain `align_to_reference` (no cross-correlation, no
integer/sub-sample split, immune to the Audit-1 leak).

**align_to_reference is genuinely buggy.** On real Silicon (known n≈3.4), self-referencing
and self_phase both recover n≈3.40/3.41, but the plain-ratio + `align_to_reference` path
gives **n≈2.30** — a real error in the time-domain alignment path (NOT the air gap, since
self-ref on the same data is correct). So for the `self_reference=False` route, prefer
`self_phase`; `align_to_reference` needs a separate fix if still wanted for the segmented
path (it also feeds the sub-sample spectral ramp). Verified: self_ref 3.402 ≈ self_phase
3.410 ≈ truth on Si s-pol.

**Validation.** Synthetic (test_window_selfref_workflow): self_phase REMOVES a planted
timing delay (like self_reference) but does NOT remove a zero-phase amplitude drift (which
self_reference does) — confirming it is phase-only. Plus the Si real-data check above.

**phase_correct_kk now documented/validated on plain THzData** (single-bounce reflection),
not just THzDataReflection — it only touches processing_dict keys. Transmission caveat:
its linear phase is the sample propagation delay (signal), so KK misplacement removal is
inappropriate there.
