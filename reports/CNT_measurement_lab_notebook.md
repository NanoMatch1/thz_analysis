# CNT THz measurement — lab notebook

A running, lab-book-style record of *why* these CNT (buckypaper) samples are hard to measure
accurately and how our interpretation evolves. **Nothing is erased.** When we later decide a
finding was wrong or incomplete, we mark it and point to the entry that supersedes it.

## How to maintain this file
- Each insight is a dated **finding** with an ID (`F1`, `F2`, …) and a **Status** line.
- Status is one of: `CURRENT` · `SUPERSEDED → F##` · `REVISED → F##` · `ONGOING` · `OPEN QUESTION`.
- To correct a finding: change its Status to `SUPERSEDED → F##` (do **not** delete the text), and
  add a new dated finding with the better interpretation that references the old one ("supersedes F##").
- Keep the **Working synthesis** at the top current; it's the one section allowed to be rewritten
  (its history lives in the findings log below).
- Detailed derivations live in the linked `reports/*tutorial*.md` and `explorations/` scripts — this
  file is the index + interpretation, not the maths.

---

## Working synthesis  *(last updated 2026-06-26)*

These samples are hard to measure for several compounding reasons:

1. **Bare air|CNT reflection is poorly conditioned.** CNT is a free-carrier (Drude) conductor →
   highly reflective → its reflection sits near `r = −1`, where the `r → n` inversion is singular.
   Low frequency is worst (a conductor is *most* reflective at DC → least contrast → least info). [F1, F2]
2. **Alignment is near-impossible on bare CNT.** We can't align the paper with the visible
   generation beam (place-it-flat-and-hope). The paper's non-flatness is **large-scale undulation**,
   which acts as a **wavefront tilt** → beam steering → a phase/coupling distortion, *not* an
   intensity loss. [F3, F4, F5, F6]
3. **The SiO₂ (or Si) window geometry fixes the conditioning** by coupling through a high-index
   medium (ATR/immersion: r moves off the −1 pole, low-frequency reopens). Si > SiO₂ > air. [F7]
4. **…but the window geometry reintroduces the air gap**, a fragile opposite-sign two-beam
   interference that drags `n` below 1 and *also* throws away the conditioning benefit unless the
   contact is gap-free. A robust de-embed is still in progress. [F8, F9, F11]
5. **The pipeline itself is validated** (recovers a known n to ±0.001 on public data); the difficulty
   is the sample physics and geometry, not the analysis code. [F12]

**Path forward (working plan):** a flat **high-resistivity Si** substrate, measuring reflection
*through* the Si onto a CNT pressed/deposited directly on it — aiming to get alignment + conditioning
+ gap-free contact at once, with a simultaneous alignment reference to the flat plane. [F13]

---

## Findings log

### F1 — Bare air|CNT reflection is poorly conditioned *(2026-06-25)*
**Status: CURRENT.** CNT is a Drude conductor → high reflectivity → `r ≈ −1`. The closed-form
Fresnel inversion `q = N₁cosθ(1−r)/(1+r)` has a **pole at r = −1**, so near `|r| = 1` a small error
in `r` produces a large error in `n` (ill-conditioned). Worst at low frequency (Drude peak at ω=0 →
most reflective there). Detail: `reports/lineshape_and_inversion_tutorial.md` §4,
`reports/reflection_interface_theory_tutorial.md`.

### F2 — The singularity has a *neighborhood*; |H| need not cross 1 *(2026-06-26)*
**Status: CURRENT** (refines F1). Distance to the pole is `|1+r| = |1−H|`, which depends on **both
|H| and arg(H)**, not magnitude alone. A sample with intensity *down* (|H| < 1, "no crossing") still
spikes `n` wherever `arg(H)` brings H real-positive, because `|1−H| → 1−|H|` there. On the CNT/gold
test pair: |H|max 0.95, yet n spiked to ~11 at 1/2/3 THz. The `min_one_plus_r` mask floor doesn't
catch these (|1+r|≈0.12 > 0.1 floor) — they're the pole's *neighborhood*, not the pole.

### F3 — Alignment is near-impossible on bare CNT *(2026-06-26, Samuel)*
**Status: CURRENT.** The CNT paper cannot be aligned with the visible generation beam the way the
reference and everything else is. Procedure is "flatten as much as possible, place, and hope," so
the sample face plane is uncontrolled run-to-run. This is the dominant *practical* error source.

### F4 — The CNT "roughness" is large-scale undulation, not within-spot RMS *(2026-06-25)*
**Status: CURRENT.** Quantified: a 50–100 µm height step gives a round-trip delay 0.24–0.67 ps
(≳ the ~0.2 ps pulse FWHM), and 50 µm *RMS within the spot* would Debye-Waller-crush |r| to 0.33 @
1 THz / ~0 @ 2 THz and broaden the pulse. We see **neither** (CNT reflects ~like gold, clean pulse).
⇒ the within-spot fine RMS is small (≲10–30 µm); the 50–100 µm is **large-scale undulation**
(correlation length ≫ λ) = locally flat-but-tilted = beam steering, which **preserves intensity and
arrival time but distorts the wavefront**. Detail: `reports/misalignment_lineshape_report.md`,
memory `cnt-roughness-two-geometries`.

### F5 — Wavefront distortion enters as a PHASE / group-delay effect *(2026-06-26)*
**Status: CURRENT.** The misalignment distorts the *pulse shape* (asymmetric), which gives the
transfer function H a **group delay** even when the sample and reference **peaks are at the same
time** (verified: 0 fs peak offset, yet arg(H) had a ~1 ps slope). That group delay sweeps arg(H)
through 0 periodically → periodic pole-neighborhood n spikes [F2]. **Peak-based recentering cannot
fix it** (peak position ≠ group delay for a distorted pulse). A linear-phase de-trend on H halves
the spikes (n 10.3→1.6) but can't fully fix it (the distortion isn't a pure delay) and is only a
*diagnostic* (it removes any real material group delay too).

### F6 — |H| > 1 is an unphysical coupling artifact (and a useful alarm) *(2026-06-25)*
**Status: CURRENT.** A passive CNT cannot out-reflect a gold mirror, so |H| > 1 means the **reference
coupled worse than the sample** (wavefront/alignment mismatch). It is the actionable alignment flag;
the pipeline now warns on it. Same effect drives the misalignment lineshape distortion in |H|.

### F7 — High-index coupling fixes the conditioning (ATR/immersion) *(2026-06-26)*
**Status: CURRENT.** Coupling the THz through a high-index medium lowers the incident impedance
(`Z ∝ 1/N`) toward the conductor's, so the mismatch is gentler and `r` moves **off the −1 pole** →
better conditioning + recovered low-frequency. Modelled: |1+r| at 0.3 THz grows air 0.06 → SiO₂ 0.12
→ Si 0.20 (~3×). **Si (n=3.4) > SiO₂ (1.95) > air.** It is the incident *medium* that matters, NOT
the reference choice — swapping gold→SiO₂ reference while keeping the CNT in air does nothing. This
is likely why the window geometry was adopted. Detail: `reports/reflection_interface_theory_tutorial.md`,
`explorations/reflection_theory/drude_interface_model.py`.

### F8 — The air gap is a fragile opposite-sign two-beam interference *(2026-06-26)*
**Status: CURRENT.** SiO₂ | air gap | CNT gives **two** reflections: SiO₂→air (n₂>n₃ → **positive**,
≈+0.32) and a gap-delayed air→CNT (n₂<n₃ → **negative**, ≈−1). A perfect contact would give a
*single* negative SiO₂→CNT reflection; the gap splits it into a positive-then-negative pair. The
spurious **positive** SiO₂/air term is the contaminant — when gap roughness (σ_d) washes out the
negative CNT term (Debye-Waller on the interference), the positive term dominates and **drags
inverted n below 1**. A poorly-pressed *uniform* gap shifts the echo without broadening; a *rough*
gap broadens it. Detail: memory `cnt-roughness-two-geometries`, `air-gap-deembedding`.

### F9 — The air gap also DESTROYS the high-index benefit *(2026-06-26)*
**Status: CURRENT.** Because the wave crosses the gap *as air*, the CNT reflection that carries the
sample happens at an **air/CNT** interface → you revert to the air-incidence (singular, F1) case
*and* add the F8 interference. So the gap is doubly damaging, and **optical contact is non-negotiable**
for the window geometry to deliver F7.

### F10 — The "Lorentzian" peak in CNT n is an inversion artifact, not material *(2026-06-24)*
**Status: CURRENT.** The persistent ~0.5–0.6 THz peak in CNT n: (a) sits at a **k-dip** (a real
resonance has k *peak*); (b) **moves in frequency with the de-embed gap** (a real resonance is pinned
at ω₀); (c) **vanishes at the physical gap**, leaving smooth Drude-like n,k. It's the near-|r|=1
inversion singularity [F1] manufacturing a peak where the gap phase rotates r past the ill-conditioned
point. CNT is expected to be **Drude** (smooth, σ peak at 0), not Lorentzian. Detail: memory
`cnt-lorentzian-is-inversion-artifact`.

### F11 — Air-gap de-embedding: status *(2026-06-25, ONGOING)*
**Status: ONGOING.** Two forward models (Route A statistical-gap Fabry-Pérot, Route B graded EMT)
agree qualitatively (both lift n>1, agree low-f) but **neither fits well** (RMS ~30× noise → model
mismatch) and material params are unstable (single-measurement degeneracy). **Minimum-phase / KK** is
the principled separator (material phase fixed by |r| via causality; the *excess* linear phase is the
pure gap) — now viable because the improved prep is low-roughness (it was marginal at the old ~8 µm).
On good-contact data: gap ≈ 5 µm (~25 fs), and the material-group-delay contamination of the raw
pulse delay is small but real (~9%, ~2.6 fs). **Transmission is impossible** (CNT too conductive/thick),
**pressure series impractical** (no true-pressure handle, slow). Detail: memories `air-gap-deembedding`,
`cnt-gap-calibration-constraints`.

### F12 — The analysis pipeline is validated; the difficulty is physical *(2026-06-25)*
**Status: CURRENT.** The transmission pipeline recovers a **known** n to mean |Δn| ≈ 0.001 on a public
synthetic dataset (phoeniks). The only deviation is mild under-resolution of *narrow* absorption lines
from time-windowing — and CNT has no narrow lines. ⇒ the inter-measurement variation is **sample +
geometry**, not the code. (Si calibration also classified our wafers: today's = HR float-zone, Chris =
doped/conductive, Denis = a *reflection* dataset.) Detail: memory `pipeline-validated-public-dataset`.

### F13 — The fundamental tension and the path forward *(2026-06-26)*
**Status: CURRENT / plan.** Bare reflection = simple, contact-free, but ill-conditioned
(alignment-limited). High-index coupling = well-conditioned, but needs gap-free contact. They pull
toward different geometries. **Working plan:** a flat **HR-Si** base substrate, measuring reflection
*through* the Si onto a CNT pressed/deposited directly on it — aiming for alignment + conditioning +
gap-free contact together, with a simultaneous alignment reference to the flat plane (to be tested).
Design the jig around Si specifically (F7), not a generic flat plane.

### F14 — The low-frequency n taper in Si window-reflection is an artifact, not conductivity *(2026-06-26)*
**Status: CURRENT.** Samuel saw Si `n` taper toward 0 below ~0.7–1 THz in the window self-ref
geometry and wondered if it looked "conductive." It is an **instrumental/inversion artifact**, not a
material response — three reasons: (a) it is present in BOTH the clean `main_alignment_tests\Si`
(camera-align) set, which is flat n≈3.4 / k≈0.2 *above* ~0.7 THz, and the poorer
`2026-06-23 export\silicon` (colinear) set; (b) we independently know this Si is high-resistivity
(transmission + the flat region here); (c) **n→0 at low f is the *opposite* of a conductor's
signature** — a Drude conductor goes to *large* n toward DC. Mechanism: for clean Si the Fresnel
ratio |H| should be flat ~0.66, but both sets **sag below it toward low frequency** — the low-f THz
beam is large/divergent so front vs back (and sample vs reference) couple differently, the self-ref
no longer fully cancels it, |H| and the phase corrupt, and inverting that low-f r drives n→0.
⇒ **trust only ~0.7–2 THz** in this reflection geometry. Separately, the colinear set is a WORSE
measurement overall (n sloped downward, k large & oscillating, 28% |H|>1 vs 16%, FP-ripple in |H| =
worse air gap/contact), so its taper is the systematic low-f artifact *plus* degraded coupling. The
"good remembered" dataset is the **camera-align** one. Config was fine (n_sio2=1.96, half_width 4,
air_gap off, eps_background only affects σ); the SNR mask just keeps marginal low-f bins. Figure
`explorations/reflection_theory/si_lowf_taper.png`. Relates to F2/F5/F6 (coupling→phase) and F7.

### F15 — Strategy: window (Si) is the path; bare reflection is a high-f cross-check *(2026-06-27)*
**Status: CURRENT** (strategic synthesis; refines/uses F1–F14). Decision after working through the
physics with Samuel:
- **Bare and window are NOT parallel options.** The science goal is **carrier dynamics**, whose most
  diagnostic content (Drude peak shape, scattering time τ; 1/τ ~ THz) lives **at/below 1–2 THz**.
  Bare air|CNT reflection loses that band — and not as a fixable processing weakness but as
  **fundamental air-incidence conditioning on a conductor** (r on the −1 pole, F1/F2). So bare is
  *structurally mismatched* to the question: even perfectly aligned it can't deliver the needed band.
- **The bare-reflection pipeline (run_me_reflection_single, gold-ref) only behaves after stripping the
  linear phase entirely** — a fragile band-aid that also removes real material group delay (F5). This
  is the bare path's signature failure. **The SiO₂ window pipeline (run_me_reflection.py, self-ref) is
  fine** — its W=Y2/Y1 cancels the timing structurally, no linear-phase strip needed. (Earlier I wrongly
  attributed the blow-up to the window path; corrected here — Samuel.)
- **Conditioning (bare) is fundamental physics; the gap (window) is a technical contact/de-embed
  problem.** Technical problems yield to engineering; fundamental ones don't. ⇒ **commit to the window
  geometry — ideally Si (n=3.4 ≫ SiO₂ 1.95 → far more conditioning headroom, F7) — and spend the effort
  on the gap.** A flat HR-Si substrate is simultaneously the alignment plane *and* the best shot at
  gap-free contact (press/deposit directly) — one jig, three problems (F13).
- **Alignment (Samuel's untested method) fixes the rigid plane → reliability/reproducibility, NOT
  bandwidth.** It doesn't touch the air-incidence conditioning, and residual paper-surface aberrations
  (wavefront) become the new noise floor. Necessary, not sufficient, for bare.
- **Role of bare reflection:** keep as a **>2 THz cross-check** + carrier-density sanity anchor, not the
  primary instrument.
- **Code flag (run_me_reflection.py):** `n_fft=500` is likely too small for the shared-axis path (the
  windowed trace keeps full length; rfft truncates if n_fft < trace length → cuts the late pulses). The
  `minimum_fft_length` assert from run_me_low-level.py should be ported here. [check]

**Next (2026-06-27):** deep literature recanvas + verify the analytical de-embed approach (min-phase
the current candidate); Samuel tinkering the mount in parallel. See `reports/air_gap_deembed_research.md`
(in progress).

### F16 — De-embed approach refined by literature recanvas *(2026-06-27)*
**Status: CURRENT** (advances F11). Recanvassed the THz reflection / de-embed / phase-retrieval
literature → `reports/air_gap_deembed_research.md`. Conclusions:
- **Geometry is the primary fix and is field-standard: high-index coupling on HR-Si (ATR / Si-window),
  sample deposited/intimately pressed.** This is *the* established way to do conductive-film THz —
  fixes conditioning (F7) and removes the gap if deposited (F13). Build effort goes here.
- **Min-phase de-embed was the right FAMILY (causality phase retrieval) but the wrong implementation.**
  Upgrade: (a) reconstruct the causal phase with **MEM (maximum-entropy)** not Hilbert(ln|r|) — MEM is
  far better at the spectrum *edges* (our low-f weak spot) and is specifically recommended for
  conductors; (b) pin the gap's linear (timing) phase by **amplitude-KK minimisation** (the robust
  method of arXiv 2412.18662, *designed for highly reflective/conductive samples* — find the shift that
  makes |r| KK-consistent, instead of fitting the corrupted measured phase as our detrend does); (c)
  **anchor with HR-Si** (known n — both KK and MEM improve markedly with 1–2 anchor points).
- The statistical-gap Fabry-Pérot (Route A) stays as the *residual amplitude* model (Debye-Waller +
  fringes), small with good contact. `detrend_transfer_phase` / time-delay is demoted to a diagnostic.
- **Gap found NOT to be addressed in the literature for a ROUGH/distributed gap** (only uniform offset)
  — that may be our novel contribution. Worthy attempt to adapt: arXiv 2412.18662 + MEM + Si anchor.
Next code: `mem_phase_retrieval`, `gap_shift_amplitude_kk`. See research report for full refs (RQ1–RQ4).

### F17 — Mean gap d̄ is the enemy; contact-roughness spread σ_d is a secondary high-f effect *(2026-06-27, Samuel + me)*
**Status: CURRENT** (refines F8/F9). Samuel's question: if bare reflection barely notices surface
roughness, why fear roughness of *contact*? Resolution — they act through **different channels**:
- **Bare single-interface reflection is insensitive to roughness because THz is sub-wavelength to it.**
  λ ≈ 300 µm at 1 THz, so even 10–50 µm RMS gives spot phase variance `2kσcosθ ≪ 2π` →
  Debye-Waller `exp(−2(kσcosθ)²) ≈ 1`, specular reflection intact. Large-scale undulation (L≫λ) only
  *tilts* the wavefront → beam-steering/alignment (F4), not amplitude. THz simply can't scatter off
  µm-scale features.
- **The contact gap is sensitive through interference, not scattering** — opposite-sign two-beam
  etalon. Split into two terms that scale differently:
  - **Mean gap d̄** — deterministic frequency-dependent interference; *present even for a perfectly
    smooth gap (roughness not required)*. This is what drags n<1 and corrupts the phase → the
    **dominant, de-embeddable** term (target of `gap_shift_amplitude_kk`).
  - **Spread σ_d** — averaging over a distribution of etalon phases → Debye-Waller loss of *fringe
    contrast* `exp(−2(ω/c·σ_d cosθ)²)`, **∝ ω² → bites at HIGH f, negligible at the low f we care about.**
- **Why contact "feels" rougher than bare = leverage, not larger scattering.** In the gap the signal
  (`r_back`) is partly cancelled by the opposite-sign front reflection, so we work in the difference of
  two large near-equal terms → any gap-phase error is amplified *fractionally*. But that leverage hits
  **d̄ as much as σ_d** — a reason to kill the mean gap, not to fear the spread.
**Implication for the build:** prioritise estimating/removing **d̄**; treat σ_d as a residual high-f
Debye-Waller roll-off (Route A) and don't over-invest in roughness statistics. A smooth *small* gap is
mostly de-embeddable; a smooth *large* gap is the real problem.

### F18 — Validation overturns part of F16: amplitude-KK is for SUPER-fringe/misplacement, NOT the small contact gap *(2026-06-27)*
**Status: CURRENT** (corrects the F16 emphasis on amplitude-KK as the gap estimator). Built MEM
phase retrieval (`mem_phase_retrieval.py`) and the amplitude-KK gap-shift finder
(`estimate_gap_amplitude_kk`) and tested both against synthetic ground truth. Results:
- **MEM beats Hilbert ~6× at the low band EDGE on a TRUNCATED band** (the sub-1-THz region) —
  validated (`mem_phase_validation.png`). But on a WIDE clean band MEM's opposite-(high-)edge
  wobble can make it *worse* than Hilbert. MEM's win is conditional on band-limiting + a
  near-edge feature, i.e. real low-f-cutoff data. Not a universal upgrade.
- **Amplitude-KK (forward-FP magnitude match, the arXiv 2412.18662 idea adapted) has TWO failure
  modes the phase-slope estimator is immune to:** (a) **conductor sign ambiguity** — min-phase
  from |r| drops the metal's ~π (r_back≈−1); the forward model needs it (fixed by an automatic
  sign search); (b) **sub-fringe degeneracy** — the gap enters |r_meas| only via FP fringes of
  period Δf=c/(2d cosθ); for our band one fringe needs ~80 µm, so gaps below ~30 µm (our
  good-contact target, ≲13 µm) are sub-fringe → |r_meas| nearly flat in d → degenerate. A
  sub-fringe warning now fires and defers to the phase estimator.
- **Where amplitude-KK DOES earn its keep:** super-fringe gaps (≳60 µm) AND it is **immune to
  alignment/misplacement corruption** (a spurious linear phase leaves |r| unchanged), recovering
  d≈120 µm even under an 8 µm misplacement that biases the phase-slope method. That is its real
  (arXiv) regime: large residual gap + suspect alignment / cross-check.
- **Corrected production plan for the SMALL good-contact gap:** the **phase-excess estimator**
  (`estimate_gap_minimum_phase`, arg(x)−φ_minphase slope) is the right tool — immune to the
  conductor sign and works sub-fringe. Its only weakness is the material-dispersion bias in the
  slope, reduced by (i) **MEM** phase on band-limited data (edge advantage) and, more
  importantly, (ii) constraining the material with a **Drude/Drude-Smith model + HR-Si anchor**
  (curved dispersion breaks the linear-phase degeneracy, F11). The window geometry already
  removes alignment, so amplitude-KK's robustness is largely pre-solved there.
Code: `mem_phase_retrieval.py` (+7 tests), `estimate_gap_amplitude_kk`/`reconstruct_material_phase`
in `deembed_air_gap_iterative.py` (+4 tests), `phase_engine='mem'|'hilbert'` selectable. All 123
tests pass.

---

## Diagnostics & tools built for this work
- **`selfref_quality`** — flags front-spot drift via std(|C|) (window self-referencing).
- **`min_one_plus_r`** floor in `invert_nk_reflection` — masks the r=−1 blow-up; warns on |H|>1 [F6].
- **`detrend_transfer_phase`** — experimental linear-phase removal, to *see* the F5 spikes vanish.
- **Air-gap de-embed** slider + `deembed_air_gap_reflection` pipeline step [F11].
- Tutorials: `lineshape_and_inversion_tutorial.md`, `misalignment_lineshape_report.md`,
  `reflection_interface_theory_tutorial.md`; model `explorations/reflection_theory/drude_interface_model.py`.

## Open questions / next
- `OQ1` — Does a flat HR-Si jig actually deliver gap-free contact, or just move the gap? [F13]
- `OQ2` — Quantify how much residual gap collapses the F7 conditioning advantage (Si-incidence + μm gap model).
- `OQ3` — Process Denis's reflection dataset properly to settle its doping [F12].
- `OQ4` — Min-phase de-embed as the production gap estimator on the latest good-contact data [F11].
