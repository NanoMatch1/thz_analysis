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
