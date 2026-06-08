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
  - Rejected `pre_window_align_peak`: integer-sample shifts only (0.05 ps granularity) —
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
