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
**Status: CURRENT.** Ran the new de-embed methods (MEM + Hilbert phase-excess gap estimate,
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
**Status: CURRENT** (first real-data test of F22 code; anisotropy dataset). Ran CNT-21/polarization
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
