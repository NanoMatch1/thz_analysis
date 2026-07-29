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

## Working synthesis  *(last updated 2026-07-02)*

These samples are hard to measure for several compounding reasons:

1. **Bare air|CNT reflection is poorly conditioned.** CNT is a free-carrier (Drude) conductor →
   highly reflective → its reflection sits near `r = −1`, where the `r → n` inversion is singular.
   Low frequency is worst (a conductor is *most* reflective at DC → least contrast → least info). [F1, F2]
2. **The SiO₂ (or Si) window geometry fixes the conditioning** by coupling through a high-index
   medium (ATR/immersion: r moves off the −1 pole, low-frequency reopens). Si > SiO₂ > air. [F7]
   p-pol adds ~6× more pole distance on top [F21, F23].
3. **…but the window geometry reintroduces the air gap** [F8, F9]. On CNT-21 the mean gap is now
   *measured*: ~5.3 µm raw phase-excess, ~1–3 µm after the Si-anchor systematic correction —
   good contact, right at the F19 sensitivity threshold. **The de-embed mathematics is done and
   VALIDATED on both Si controls** (s and p) [F26]; gaps are per-mount [F27].
4. **The instrument, alignment and per-orientation referencing are at the ~0.5–2% level** —
   realign repeatability 0.6% in |H|, reference slot-to-slot 2%. The old "alignment is the
   limiter" reading (F20/F23) is overturned; the |H|>1 alarm was misread for window geometry
   (physically expected there) [F24, F25, F28].
5. **The remaining wall is a smooth ~10% mount-coupling systematic** (pressed paper deforms the
   coupling; |C| 0.85 vs Si's ~1.0) **degenerate with material lineshape, plus air-incidence
   conditioning behind any residual gap.** No further de-embed algorithm can separate these from
   a single mount — the limit is sample presentation, not analysis. [F28]
6. **The pipeline itself is validated** (±0.001 on public transmission data [F12]; Si at 3.418 in
   both pols in this exact reflection geometry [F25, F26]).

**Path forward (unchanged, now evidence-backed):** CNT deposited/pressed on flat **HR-Si**,
measured through the Si [F13, F16] — rigid mount keeps the coupling stable (the Si controls prove
it), deposited contact kills the gap, Si incidence maximises conditioning. Interim: p-pol
window-frame σ₁ over 0.9–2.2 THz with ±15% systematic error bars [F28].

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

### F19 — A time-INVISIBLE gap is enough; the time trace is too coarse a ruler by ~10× *(2026-06-28, Samuel + me)*
**Status: CURRENT** (reconciles F1/F2/F8 with the time-domain observation; answers "are we even
sure it's the gap?"). Samuel's challenge: the back-reflection pulses (CNT and Si) show no visible
delay; he can center peaks to ~25 fs (~5 µm), and the literature says Si needs tens-to-100 µm to
shift n — so why are we struggling? **Quantified answer (script: apply gap phase to planted Drude
CNT vs Si, re-invert):**
- Δn(CNT) from a gap: **0.3 µm (1.4 fs) → −0.07; 1 µm (4.7 fs) → −0.24; 4 µm (18.9 fs) → −0.84**
  at 0.5 THz. A **4 µm gap is 19 fs — below the 25 fs centering resolution, i.e. INVISIBLE in the
  time trace — yet wrecks low-f CNT n.** To hold Δn(CNT)<0.05 needs <1.5 fs centering — beyond any
  pulse-centering method. **The time-domain peak is ~10× too coarse a ruler to certify gap-free
  contact for a near-mirror sample; the frequency-domain inversion is the sensitive instrument.**
- Same gap barely moves Si: 4 µm → Δn(Si)=−0.06 at 0.5 THz (~13× less than CNT) — matches the
  literature "Si needs tens of µm." Pole distance |1+r|: CNT 0.26 (0.5 THz)→0.64 (4 THz) vs Si flat
  0.35, so CNT is worst-conditioned at LOW f (most metallic) — exactly where Samuel struggles.
- **Roughness paradox resolved:** the bare-reflection surface and the window-gap are the SAME paper
  topography in two geometric ROLES. Bare = micro-relief ON the mirror (sub-λ → Debye-Waller≈1 →
  invisible). Window = the SPACER of a coherent etalon (round-trip phase read directly + conditioning-
  amplified → catastrophic). THz didn't change sensitivity; the roughness changed jobs.
- **Confidence (honest):** RIGOROUS = the air|CNT inversion is ill-conditioned, so ANY small phase
  residual (gap, calibration, reference, alignment) is amplified — a time-invisible perturbation
  suffices. NOT certain = that a literal air gap is the SOLE/dominant residual. Flags against
  gap-only: (i) pressing harder barely changed the data [Samuel]; (ii) Si itself isn't perfectly
  clean (F14 low-f taper = upstream); (iii) two artifact signatures (smooth ∝ω drag = gap-like vs
  spiky ~1 THz = pole-neighbourhood). **Decisive test:** Si in the IDENTICAL window geometry — clean
  Si ⇒ setup sound, CNT mess is CNT-conditioning(+gap); messy Si ⇒ upstream. Plus frequency
  signature (∝ω = gap; fixed spikes = conditioning). Samuel will revisit Si next week with careful
  alignment; the latest Si set was firmly pressed (visible-light fringes over mm ⇒ sub-µm flatness).

### F20 — First real-data trial of MEM/phase-excess de-embed: pipeline validated by Si, CNT gaps fictitious under poor conditioning *(2026-06-28)*
**Status: REVISED → F28** (the |H|>1-based "alignment is the limiter" reading is overturned for
window geometry — |H|>1 is expected physics there; the Si findings stand). Ran the new de-embed methods (MEM + Hilbert phase-excess gap estimate,
amplitude-KK cross-check) on real window self-referenced data — CNT 0/90-deg (2026_06_19) and Si
(2026-06-23, firmly pressed). Script: `deembed_realdata_mem.py`.
- **Si CONTROL is mostly reassuring:** naive Si median n = **3.33 @0.5–1 THz (within 3% of 3.42)** →
  the pipeline, reference, geometry, and self-referencing are FUNDAMENTALLY CORRECT (not a gross
  upstream error). Si droops to **2.61 @1–2 THz** — a HIGH-f artifact whose ∝ω signature matches a
  small few-µm gap (F19); Si gap estimates 2–5 µm, engine-consistent, one slightly negative (≈0±5 µm,
  matches the time-domain). Si |H|>1 in only 18–30% of bins; |C|≈1.00/0.95 (well aligned).
- **CNT is NOT trustworthy on this set:** naive n<1 (min ≈0.44), **|H|>1 in 76–90% of bins**,
  |C|≈1.07–1.09 (poorly aligned). The de-embed mechanically lifts n>1 but into UNPHYSICAL territory
  (n peaks ~11 Hilbert / ~6 MEM near 0.6 THz = pole-neighbourhood blow-up). Gap estimates **Hilbert
  22 µm vs MEM 31 µm — disagree ~35% and are ~6× larger than the <5 µm the time trace allows** →
  the phase-slope is absorbing the metal's dispersion + the |H|>1 mess into a FICTITIOUS gap. Confirms
  F19: under conditioning dominance the de-embed "gap" ≠ physical gap.
- **Verdict:** methods are READY (validated on synthetic, run on real in one command); the limiter is
  DATA QUALITY/alignment, exactly as Samuel suspected. **Next diagnostic = get Si flat at 3.42 across
  the band** (chase the 3.33→2.61 high-f droop: gap ⇒ ∝ω, flattens with firmer press/de-embed; SNR ⇒
  won't). Only then is CNT worth re-attempting. CNT 90 vs 0 indistinguishable here — both swamped by
  |H|>1; alignment is the gate. Samuel collecting Si again next week with careful alignment.

### F21 — Dual-polarization (s/p) ellipsometry is a strong route around BOTH problems *(2026-06-29, Samuel's idea + validation)*
**Status: CURRENT.** Samuel: the spintronic emitter makes polarization a free knob (rotate magnet;
rotate EO crystal to match). Idea: p-pol lowers reflectivity at the parasitic interface → pinning
points + reduced conditioning; a polarization series as an alternative to a pressure series.
**Assessment (numbers from our geometry, validated synthetically — `polarization_ellipsometry_validation.py`):**
- **p-pol attacks BOTH core problems at once.** (1) Front suppression: the parasitic SiO₂→air front
  reflection drops |r| 0.44(s)→0.19(p) at our 45°, →**0 at the Brewster external angle ~63°** →
  removes the gap interference (F8) at its source. (2) Conditioning: the air→CNT back |r| falls
  ~0.85(s)→~0.55(p) → off the r=−1 mirror (F1/F2). Demo: a 4 µm gap pushes recovered n by median
  1.26 (s) vs 0.92 (p) at 45°, and **0.16 (p) at Brewster** vs 1.13 (s).
- **Windowed self-referencing needs NO s/p channel calibration** (Samuel's question): each pol's H
  self-references against its own front pulse, so the channel response cancels inside H; only the
  BARE reflection would need channel calibration. Intensity normalises out; noise is just SNR.
- **The ellipsometric ratio ρ=r_p/r_s** is exact for a single interface (verified to 1e-6) but the
  window+gap is NOT a single interface → the naive closed form is geometry-biased. **The production
  method is a joint forward-model fit of a Drude(+anchor) n(ω) AND the gap d to s and p together**;
  it recovers planted n and d EXACTLY with the gap present (Demo 3).
- **Honest scope:** on noiseless data with a Drude model, single-pol ALREADY recovers n+d (no
  clean-data degeneracy). **Dual-pol's real payoff is conditioning/robustness under noise**: at 1%
  reflection noise, fitted-n error dual s+p 0.004 < p-only 0.009 < s-only 0.013 (Demo 4). So the win
  is (a) p/Brewster front-suppression + better conditioning, (b) over-determination vs noise/model
  error — NOT a clean-data degeneracy break.
- **Polarization vs pressure series:** NOT equivalent — polarization is superior here (easy magnet
  rotation; doesn't disturb alignment, unlike pressure which drifts hard samples; p-pol actively
  *suppresses* the gap rather than varying it; s/p ratio is reference-immune). Pressure's only unique
  power is physically shrinking the gap.
- **Caveat:** anisotropic (aligned-CNT) buckypaper can s↔p cross-couple → breaks the scalar-ρ picture
  (clean for isotropic Si); could become a *measurement* of the anisotropy (cf the 0/90° rotations).
**Recommended first test (current 45°): measure s AND p on Si and CNT** → predictions: p-pol Si shows
less high-f droop (F20); CNT |H|>1 fraction drops from 76–90%. Then raise incidence toward ~63° to
null the front. 5 tests, 128 total pass.

### F22 — p-pol inversion + dual-pol joint fit ported to the pipeline *(2026-07-01)*
**Status: CURRENT** (implements F21). Ported the dual-polarization methods from the exploration into
core + pipeline, ready for the s/p CNT data being collected:
- **p-pol reflection inversion in `thz_core.invert.invert_nk_reflection`** (was `NotImplementedError`).
  Closed-form quadratic in N₂²; the genuine p-pol root AMBIGUITY (two indices reproduce r_p to
  machine precision — physical vs its "gain twin") is resolved by a PASSIVITY penalty (pick Im(N₂)<0,
  k>0), residual breaking ties for near-lossless. Round-trips to 1e-15 across 0/45/63°/window.
  `_resolve_reflection_geometry` made polarisation-aware (window front reference uses fresnel_p for p).
  **`run_me_low-level.py` now handles a p-pol dataset by just setting `config['geometry']['polarization']='p'`** — no other change.
- **Dual-pol joint fit** = new `thz_core.reflection_gap`: pluggable material model (`drude_index`
  default), forward model (`single_gap_reflection`, `gap_round_trip_phase`), and
  `fit_reflection_gap_dual_pol` — fits ONE n(ω) + one shared gap d to s AND p at once. Recovers
  planted n+d to 1e-15; noise test confirms dual < s-only error. Gap fit freely OR `fixed_gap_um`
  (Si-anchor). Adapter wrapper `thz.fit_dual_pol_reflection(ds_s, ds_p, config)` pairs samples by
  normalised name and pulls the stored `reflection_r`/`r_reference`. Driver `run_me_dual_pol.py`
  processes both dirs and joins. NOTE the gap angle = the EXTERNAL angle (gap is air → Snell returns
  the beam to it).
- Low-level API (Samuel's request): `core.invert_nk_reflection(..., polarization='p')` and
  `core.fit_reflection_gap_dual_pol(...)` are DataSet-free and unit-tested (12 new tests, 140 total).
**Reminder of the F21 predictions to check on the new data:** p-pol Si droops less than s (gap
suppressed); CNT |H|>1 fraction drops from 76–90% (s). Set the two dirs in `run_me_dual_pol.py`.

### F23 — First real dual-pol run (CNT-21): p-pol conditioning confirmed (~6×), but alignment still the limiter *(2026-07-01)*
**Status: REVISED → F24/F27/F28** (reference pairing was wrong — the second label is sample
orientation, not detection pol [F24]; |H|>1 and |C| were misread as alignment damage [F28]; the
joint-fit railing persists even with correct pairing because the gap is per-mount [F27]. The
p-pol conditioning result stands). Ran CNT-21/polarization
(s-pol/ + p-pol/ folders, full {S,P}×{fiber∥,⊥} matrix) through the ported p-pol inversion + dual-pol
joint fit. Script: `explorations/air_gap_cnt_reflection/process_cnt21_polarization.py`.
- **Anisotropy pairing:** fiber 0∥S, 90∥P → 0-0=S∥→n_∥, 0-90=S⊥→n_⊥, 90-0=P⊥→n_⊥, 90-90=P∥→n_∥.
  Valid joint-fit pairs (same axis seen by s AND p): **n_∥ = (0-0 ⊕ 90-90)**, **n_⊥ = (0-90 ⊕ 90-0)**.
  Each folder has co-pol + cross-pol references (SS/SP/PS/PP = source-detect); paired each sample to
  the CO-POL reference (SS for S, PP for P).
- **p-pol conditioning CONFIRMED (~6×), the F21 core physics.** Pole distance |1+r| (distance to the
  r=-1 mirror, the TRUE conditioning metric): **s-pol 0.30–0.40 vs p-pol 1.81–1.90.** p-pol moves the
  reflection far off the mirror pole — largely via the SIGN of the p-pol front reference (r_front:
  s=+0.44, p=-0.195). This is the real, physical p-pol win.
- **|H|>1 did NOT drop with p-pol (85–98% both)** — but that was the WRONG metric: |H|>1 is a
  common-mode ALIGNMENT/coupling artifact (sample vs reference front-spot), which p-pol is not expected
  to fix. |C| medians 0.69–0.93 (structured; 0-90 flagged) → **alignment is still the limiter (F20).**
- **Joint Drude+gap fit RUNS on real data** but **rails at parameter bounds** (plasma=20 THz cap both
  axes; eps_inf=12 cap for ⊥) with **poor residual RMS ~0.16** (~20–30% of |r|) → magnitudes NOT
  trustworthy (single Drude fighting alignment artifacts + an ill-conditioned s-channel dragging the
  joint fit). **BUT the anisotropy has the physically-correct SIGN**: n_∥,k_∥ > n_⊥,k_⊥ (more metallic
  along the fibers) — encouraging qualitative result.
- **Code hardening this run:** core `fit_reflection_gap_dual_pol` now drops non-finite + DC (f≤0) bins
  (Drude diverges at DC); windowed half-width reduced to 2 ps (first pulse sits 2.2 ps from trace start
  → 4 ps clipped Y1 and corrupted the self-reference).
- **Takeaways:** (1) the p-pol + dual-pol machinery works end-to-end on real data; (2) p-pol delivers
  its conditioning promise (~6× off the pole); (3) alignment (|C|, |H|>1) remains the gate, as for Si
  (F20); (4) next: better alignment + Brewster incidence (~63°) to compound the p-pol benefit, and
  consider p-weighted or p-only inversion since the s-channel is far worse conditioned; widen plasma
  bound only once alignment is fixed. Si benchmark (tomorrow) still the decisive setup check.

### F24 — Naming convention corrected: the second label is SAMPLE ORIENTATION; one reference per channel *(2026-07-02, Samuel)*
**Status: CURRENT** (corrects the pairing used in F23). Samuel: in `X-Y`, X = THz polarization
w.r.t. the plane of reflection (0=S, 90=P) and **Y = sample/mount orientation** (0 = fibers
vertical, 90 = rotated) — NOT detection polarization (the detection crystal + gate rotate WITH
the THz pol, so their shared frame is constant; pol is defined against the reflection plane).
The SiO₂ window faces are **not parallel**, so rotating the mount precesses the front reflection
around the frequency-anisotropic EO detection cone → **each sample must pair with the reference
recorded at its own pol + orientation slot** (`sample_X-Y ↔ reference_X-Y`; the SS/SP/PS/PP tags
are pol+slot). F23 paired both S samples to SS and both P samples to PP — **mispaired for
0-90 and 90-0**. Measured cost: 0-90 |C| median 0.686 (mispaired) → 0.968 (correct); 90-0 nearly
insensitive (0.860 → 0.850, the PS/PP references are similar). Channel-paired processing:
`explorations/air_gap_cnt_reflection/process_cnt21_channels.py` (each pair as its own
mini-dataset; per-channel `.npz` in `cnt21_channel_results/`).

### F25 — Si-p control: a sparse (two-window) time axis silently collapsed the inter-pulse delay; after repair, the p-pol inversion is VALIDATED on real data *(2026-07-02)*
**Status: CURRENT.** The new FZ-Si p-pol control initially inverted to nonsense (n_med 0.93,
|H| spiraling +237 rad across the band = a spurious **+17.26 ps** linear phase, with |H| flat
and CORRECT at 1.271 vs Fresnel 1.273). Root cause: the acquisition recorded **two time windows
(153.00–159.65 + 177.00–184.45 ps) and the .acc omits the 17.35 ps between them**; the pipeline
assumes uniform dt, so the FFT collapsed the inter-pulse delay 24.85 → 7.56 ps. Zero-filling the
unrecorded gap onto the uniform grid (`repair_sparse_time_axis`) fixes it exactly.
- **After repair: Si-p n_med = 3.38 (target 3.418), k ≈ 0.2** → the F22 p-pol closed-form
  inversion + PS-reference pairing + geometry are validated end-to-end on real data. Si-s-old
  gives n_med = 3.42 (s path re-validated on this window too).
- **⚠ pipeline gap: `build_full_trace_reflection` should ASSERT dt uniformity** — this failure
  was silent and produced plausible-looking garbage. (Recommended guard; not yet implemented.)
- **p-pol low-frequency artifact identified:** every p channel shows n → 0.7074 = n₁·sinθ (the
  GRAZING-limit root) below ~0.7–0.9 THz — the p-pol root-picker collapses to the grazing root
  where the phase is corrupted (gap + noise). It is an inversion artifact with a recognisable
  signature (n pinned at exactly n₁ sinθ₁), not material. **[FIXED in F29 — was a per-bin
  passivity picker choosing the wrong branch; now a global branch vote.]**

### F26 — De-embed chain validated on BOTH Si controls; the p-pol closed form is fragile to small gap-phase errors *(2026-07-02)*
**Status: CURRENT** (advances F18/F20; the F20 "Si next week" test is done and passed).
Script: `deembed_si_controls.py`. Oracle = the gap d that best flattens n onto 3.418.
- **Si-s-old:** phase-excess estimates d = 0.91 (MEM) / 0.95 µm (Hilbert), oracle 1.5 µm;
  de-embedding with the estimate flattens n to **median |n−3.418| = 0.058** across 0.5–2 THz.
  The full chain (x-extraction → phase-excess d → strip 2β → air-frame re-inversion) WORKS on
  real data.
- **Si-p:** the algebraic x-extraction alone (d = 0) gives **flat n = 3.418 ± 0.04 over the whole
  band** — even repairing the low-f grazing-root region. Oracle d = 0.0 µm. But the phase-excess
  estimators report +2.0 (MEM) / +2.8 µm (Hilbert) — a **systematic ≈ +2.4 µm** — and applying it
  **root-flips the p-pol inversion catastrophically** (median |n−3.418| → 2.7). Two lessons:
  (a) ±1 sample of timing registration (25 fs) ≡ **2.65 µm of apparent gap** — the estimator bias
  is exactly at that scale, so **the Si control in the same slot calibrates it**; (b) **p-pol
  trades the r = −1 pole for a root-swap boundary**: small gap-phase errors flip the branch, so
  for p-pol the gap must be pinned externally (Si anchor / s-channel), never trusted from a
  phase-slope fit alone, and closed-form p inversion needs a continuity/physicality prior before
  production use.

### F27 — CNT-21 gap numbers; per-mount gaps break the shared-gap dual-pol assumption *(2026-07-02)*
**Status: CURRENT** (constrains how F21/F22 dual-pol can be used).
- Phase-excess (MEM) gaps, raw: **0-0: 3.4, 0-90: 5.6, 90-0: 5.8, 90-0R: 5.3, 90-90: 5.3 µm** —
  consistent to ±0.4 µm across engines and across the realign repeat. Si-anchored (−2.4 µm
  systematic, F26): **physical gap ≈ 1–3.4 µm**, i.e. genuinely good contact (cf. F19's 4 µm
  wrecking threshold — we are AT the sensitivity limit, which is why the artifacts persist).
- **The gap is per-MOUNT:** 0-0 (3.4) vs 90-90 (5.3 µm) differ well outside noise. Each
  orientation is a separate pressing. ⇒ **the dual-pol joint fit's "one shared gap" assumption is
  structurally wrong for anisotropic samples**: to see the same fiber axis with both pols you must
  rotate the sample with the pol → re-mount → new gap. The joint fit needs per-channel gaps (or
  simultaneous s+p acquisition without re-mounting, which anisotropy forbids). This — not the F23
  mispairing — is a core reason the joint fits rail: corrected pairing + wide bounds + Drude-Smith
  + per-channel nuisance ALL still leave rms ≈ 0.08–0.13.

### F28 — Error budget closed: the instrument is at ~0.5–2%, every model stalls at ~10% SMOOTH mismatch; verdict = analysis is at its information limit, the binding constraint is mount-coupling + conditioning *(2026-07-02)*
**Status: CURRENT — the requested verdict.** Scripts: `fit_cnt_dual_pol_corrected.py`,
`fit_cnt_drude_smith.py`, `fit_cnt_nuisance_scale.py`, `residual_structure_analysis.py`.
- **The measurement is EXCELLENT:** 90-0 vs 90-0R (independent realignment) agree to **0.6% in
  |H|, 1.5 fs in timing, 0.003 rad in detrended phase**. Reference slot-to-slot W ratios (same
  window, 4 mounts): |ratio| flat to 2%, phase structure 2–4 mrad. Smooth-detrended H structure
  only 0.6–1.1%. **F20/F23's "alignment is the limiter" is OVERTURNED for this dataset** — the
  alignment, referencing scheme and repeatability are all at the % level.
- **|H| > 1 metric retired for window geometry:** the window reference back-reflection is weak
  (s: +0.44, p: −0.196), so any sample reflecting more gives |H| > 1 **physically** (p-pol Si:
  expected H = −1.27, measured |H| = 1.271!). The 76–100% |H|>1 fractions read as alignment damage
  in F20/F23 were largely EXPECTED physics. |H|>1 remains meaningful only where the expected
  |H| < 1 (e.g. Si in s-pol, gold-referenced bare geometry).
- **But every material+gap model leaves a smooth ~9–13% residual** (plain Drude, Drude-Smith
  (c rails at −1), free/fixed/Si-anchored gap, flat per-channel scale+delay nuisance — all
  ≈ rms 0.08–0.13 vs the 0.005 reproducibility floor). The mismatch is SMOOTH (not fringes — no
  echo delay signature), so it is either genuinely non-Drude(-Smith) material response or a
  smooth frequency-dependent coupling bias between the sample and reference MOUNTS. The flat
  |C|: CNT mounts 0.85 vs Si-p 0.966 / Si-s 0.996 fingers the pressed paper deforming the
  window/mount coupling; C cannot correct it (the front/back split is unknowable — audited), only
  flag it.
- **What survives robustly today:** (1) the mean gap is measured: ~5.3 µm raw / ~1–3 µm
  Si-anchored, reproducible; (2) **anisotropy is robust in every variant**: σ_∥ > σ_⊥ (roughly
  2–7×) with n_∥, k_∥ > n_⊥, k_⊥; (3) the p-pol window-frame trusted band (~0.9–2.2 THz) is the
  best single-spectrum view (Samuel's observation confirmed; below ~0.9 THz the grazing-root +
  gap artifacts take over); (4) σ₁ in that band is surprisingly robust to the coupling-scale
  systematic: ±10% |H| scale → σ₁ = 52–62 S/cm (90-0, 1–2 THz) while n/k individually swing ±25%.
- **VERDICT:** the de-embedding mathematics is done and validated (F26); no further analytical
  de-embed will buy accuracy here because the remaining error is (i) a smooth mount-coupling
  systematic the data cannot self-separate from material response, and (ii) closed-form
  inversion conditioning at air incidence behind ANY residual gap (F9/F19) — both physical/
  geometric, not algorithmic. **The limiting factor is sample presentation, not analysis.**
  The F13/F16 route (CNT deposited or pressed on rigid flat HR-Si, measured through the Si)
  attacks all three at once: rigid substrate → mount coupling stable (Si controls prove rigid
  mounts stay at |C| ≈ 1); deposited contact → gap → 0; Si incidence → conditioning headroom.
  Interim recipe on existing data: p-pol window-frame σ₁ over 0.9–2.2 THz with ±15% systematic
  error bars + phase-excess gap monitoring per mount + Si-slot control each session.

### F29 — Two pipeline bugs fixed: the p-pol "hard cut to zero" (grazing-root collapse) and the air-gap de-embed "slams to 0.7" *(2026-07-02, Samuel flagged both)*
**Status: CURRENT** (fixes the F25 p-pol grazing artifact; corrects `deembed_air_gap_reflection`).
Samuel spotted two data-analysis artifacts and both were real bugs.
- **(1) The p-pol low-f "hard cut and drop to ~0" is the grazing-root collapse (F25), now FIXED.**
  Not a mask, not zero — n dropping to **n₁ sinθ = sin45° ≈ 0.707**. The p-pol Fresnel inversion
  is a quadratic in N₂² with two exact roots — the physical index (u+, e.g. Si 3.35 or the CNT
  metal branch) and a spurious low-index twin (u-, degenerates to the grazing value for a highly
  reflective sample) — and BOTH reproduce r_p to ~1e-15, so residual can't separate them. The old
  picker used a **per-bin passivity penalty** (reject Im(N)>0). But wherever the sample is weakly
  absorbing, the sign of Im(N) for the physical root follows the sign of Im(r), which the corrupted
  low-f measurement (gap phase, |H|>1 coupling) **flips** — so the penalty killed the correct root
  and chose the twin. The cutoff frequency = wherever Im(r) crosses zero → **0.87 THz for Si, ~1 THz
  CNT∥, ~0.7 THz CNT⊥** (Samuel's exact numbers; different per dataset because the phase differs).
  Confirmed on the well-conditioned Si control (|1+r|≈1.24, so NOT a conditioning failure — a
  branch-selection failure). **Fix:** replaced the per-bin rule in `thz_core.invert._invert_p_pol_index`
  with a **global branch vote** — pick u+ vs u- by which is passive (Im≤0) across the whole band,
  robust to minority band-edge sign flips; grazing kept only as the m→0 fallback. Validated: Si
  holds n = 3.0–3.42 across 0.3–2 THz (was collapsing <0.88 THz); CNT-90-90 returns the physical
  metal branch (n 4.6–31, k 14–27, Drude-like; the n≈31 near 0.9 THz is the separate F2 pole
  neighbourhood, maskable via `min_one_plus_r`) instead of grazing everywhere. Regression test
  reproduces the exact failure (old picker collapses 50% of bins, vote 0%).
- **(2) The air-gap de-embed "slams n to 0.7 regardless of spacer, even at position=width=0" had a
  real bug + a design trap.** (a) **Bug:** `deembed_air_gap_reflection` hardcoded `polarization='s'`
  in its re-inversion, ignoring the p-pol config. (b) **Trap:** it *unconditionally* stripped the
  SiO₂→air front (Möbius x=(r−r_front)/(1−r_front·r)) and re-inverted at **AIR incidence** even at
  zero gap — so it was never a no-op. For a gap-free sample this manufactures a small wrong-sign x
  (Si: x≈+0.42, but a real air→Si reflection is negative) and inverting a weak x at air incidence
  gives n=√(q²+sin²45°)→√0.5≈0.707 (the grazing floor); pushing position_um rotates x by e^{+2iβ},
  walking n toward 1. This is **F9 in action** — behind any air gap the inversion reverts to the
  ill-conditioned air-incidence case; applied where there's no gap it collapses. **Fixes:** read
  polarization from config; make position_um=width_um=0 a genuine **no-op** (leave window n,k
  untouched — a zero gap is not a gap). The de-embed's p-pol re-inversion also inherits the (1) fix.
- **(3) Same air-incidence re-inversion bug in the interactive air-gap slider** (`air_gap_slider_
  explorer.py`): `invert_air_incidence` also defaulted to `polarization='s'`. Threaded the
  measurement polarisation onto each sample record (read from the dataset/PIPELINE config in
  `samples_from_dataset` / `load_measured_samples`), so `compute_curves` re-inverts in the correct
  branch. Validated: on p-pol data the slider now returns the physical metal branch for CNT-90-90
  (n 5.4–21, k 12–16) across the physical 0–3 µm gap range (cf. F27's ~1–3 µm), instead of grazing
  everywhere; Si at gap=0 gives 3.38 (was 0.76). It still collapses at over-large trial gaps (≳5 µm
  on CNT, any gap on gapless Si) — that is the de-embed over-stripping into the air-incidence
  degeneracy (F9), now a useful "you've gone too far" signal rather than a permanent collapse.
- **Files:** `thz_core/thz_core/invert.py` (`_invert_p_pol_index` + `_p_pol_index_candidates`
  helper), `dataset_core/adapters/thz_adapter.py` (`deembed_air_gap_reflection`),
  `explorations/air_gap_cnt_reflection/air_gap_slider_explorer.py`. Tests: 343 pass (147 outer incl.
  new grazing regression + updated zero-gap-noop; 196 nested incl. the F22 p-pol round-trip that
  replaced the stale `not_implemented` assertion).

### F30 — p-pol grazing-root collapse is triggered BY removing the front-pulse timing offset, not by a residual one; fixed in the root picker *(2026-07-09)*
**Status: CURRENT** (extends F25/F26/F29). New MINTS 2026-07-07 doped **n-type** Si control
(pressed in the SiO₂ window, p-pol) inverted fine on the PLAIN back ratio (n≈3.14–3.20) but
railed to the grazing root **n = n₁·sinθ = 0.707** the moment `self_phase` (front-pulse phase
referencing) was on. A previous FZ (high-resistivity) Si set worked in all modes.
- **Mechanism (diagnosed, reproduced headlessly).** `self_phase` forms `H=(Y2_s/Y2_r)·(C/|C|)`,
  `C=Y1_r/Y1_s` — it multiplies H by the FIRST(front)-reflection phase difference between
  sample and reference. The MINTS sample & reference are separate acquisitions with a ~47 fs
  front-pulse timing drift (arg(C) linear-through-origin ⇒ a *legitimate* offset). Removing it
  rotates arg(H) onto ≈−π (Si is higher-index than SiO₂ ⇒ real-negative r), i.e. **Im(r) crosses
  0**. Being near-lossless AND high-index, the sample sits ON the p-pol root-swap boundary (F26):
  the physical high-index root then reads as slight gain (Im>0) and the F29 global passivity vote
  collapses the whole band onto the passive **grazing twin**.
- **Samuel's fix hypothesis (eat the ~47 fs≈1-sample delay by a clean integer time-shift + crop,
  via a new `align_to_reference(crop_to_overlap=True)`) was DISPROVEN.** A time-shift sweep on the
  sample spectra shows the flip happens *exactly when the offset is fully eaten* (−47/−50 fs →
  0.723), identical to `self_phase`; the plain ratio only survives because the *uncorrected* 47 fs
  keeps arg(H) a hair short of π. **Removing the offset (by ANY method — self_phase, cross-corr,
  crop) is the trigger, not the cure.** The physical high-index root (n≈3.3) is PRESENT in the
  corrected H but unpicked; the twin degenerates to n₁sinθ.
- **Real fix = the root picker, not the timing stage.** `thz_core.invert._invert_p_pol_index`
  now applies a **physicality override**: n₁·sinθ is the evanescent/TIR floor, so if the passivity
  vote lands at/below it while the other quadratic branch is genuinely high-index, take the
  high-index branch (even if it reads slight gain k<0 — an honest "phase over-rotated / residual
  gap" flag, far better than 0.707). Additive: it never overrides a vote that already picked
  high-index, so every conductor / low-loss-dielectric round trip is unchanged. MINTS self_phase
  now returns **n≈3.30**; FZ unchanged (3.44). Regression test
  `test_p_pol_grazing_override_when_physical_root_reads_as_gain` (synthetic reproduces the exact
  collapse: old vote picks grazing twin, override holds n=3.28). Tests: dataset_core 204,
  nested thz_core 193 (+invert 24) pass.
- **New diagnostic** `thz.plot_first_reflection_phase_diagnostic` (adapter, wired into
  run_me_low-level behind `general.selfref_phase_diagnostic`): shows front(Y1) & back(Y2)
  magnitudes, arg(C) with a timing-fit, arg(H) plain-vs-self_phase, **Im(r) with the Im(r)=0
  root-swap line (the smoking gun)**, |H|, and n both ways vs grazing/Si guides; prints a
  `GRAZING-ROOT FLIP` flag. Note: |C| and `selfref_quality` (std|C|) MISS this — the front spots
  match in amplitude (|C|≈0.96 both sets); the signal is entirely in front-pulse PHASE.
- **Caveat for the operator:** with the picker fixed, `self_phase` returns the physical branch —
  but a k<0 output is the flag that arg(H) is slightly over-rotated (residual gap / imperfect
  timing). Use ONE timing correction (self_phase alone on the shared axis is cleanest), not
  self_phase + align together.

---

## Diagnostics & tools built for this work
- **`plot_first_reflection_phase_diagnostic`** — front-reflection phase + the self_phase p-pol
  grazing-root-flip check (Im(r)=0 root-swap panel) [F30].
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
