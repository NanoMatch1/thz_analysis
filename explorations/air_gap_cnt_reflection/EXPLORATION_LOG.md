# Exploration log — air-gap de-embedding for CNT window-reflection

A running record of the explorations in this folder, each as **Problem → Hypothesis →
Approach → Result → Conclusion/Recommendation**. Read top-to-bottom for the story; each
entry stands alone for reference.

## The overarching problem

Geometry: a thick, opaque, highly conductive CNT bundle pressed against the flat back face
of a SiO2 window, probed in reflection at 45° (s-pol), self-referenced against a SiO2-only
trace. We want the CNT optical constants n, k (and conductivity σ).

The symptom that started this series: the extracted refractive index **n droops below 1**
across most of the band — unphysical for a conductor. The shape of n(ω), k(ω), σ(ω) looks
sensible, but the absolute n is wrong.

Related artefacts: `reports/cnt_rough_gap_literature_canvas.md`,
`reports/gap_roughness_modelling_implementation_plan.md`, ANALYSIS_NOTES §9/§9b.
Run scripts from the repo root, e.g.
`PYTHONPATH=. .venv/Scripts/python.exe explorations/air_gap_cnt_reflection/air_gap_models.py`.

---

## 1. `inspect_cnt_reflection_traces` — locate the reflections in the new data

- **Problem.** The 2026_06_19 measurement uses a *thicker* SiO2 window and a new holder, so
  the front/back reflection timings differ from the old presets.
- **Approach.** Scan-average each multi-scan `.acc`, plot, find the two reflection peaks.
- **Result.** First reflection (front face) ≈ **155.85 ps**, second (back face / sample)
  ≈ **180.8 ps**, ~25 ps apart. CNT-0° and CNT-90° overlap; the reference differs only at
  the second reflection — the expected self-referencing signature.
- **Conclusion.** Set regions first (153.5, 159.0) ps, second (178.5, 183.5) ps for the rest
  of the series.

## 2. `compare_reflection_pathways` — does better data fix the droop?

- **Problem.** With cleaner data and a more consistent alignment, does n still fall below 1?
  And do the two processing pathways agree?
- **Hypothesis.** The droop might be a windowing/processing artefact that better data fixes.
- **Approach.** Run the same data through BOTH pathways — shared-axis (self-ref ON) and
  segmented (self-ref OFF) — aligned on the first reflection, sub-sample correction carried
  through. Plot n, k, Re σ, Im σ for both CNT orientations.
- **Result.** n **still** drops below 1 over ~86–93% of the band (shared n≈0.5–0.6,
  segmented n≈0.7–0.8; they converge above ~1.5 THz). k agrees well between pathways;
  conductivity is Drude–Smith-like; CNT-0 vs CNT-90 show clear anisotropy.
- **Conclusion.** The droop is **not** a windowing or pathway artefact — it survives both
  paths and the new data. The cause is physical → points at the contact air gap. Anisotropy
  is real and must be fit per orientation.

## 3. `explore_air_gap_deembedding` — is a contact air gap the cause? (synthetic)

- **Problem.** Why does n fall below 1, and can it be undone?
- **Hypothesis.** The CNT does not contact the window perfectly: a thin **air gap** makes
  the interface a 3-medium stack SiO2 | air (d) | CNT — a Fabry–Pérot **etalon**, not the
  single SiO2→CNT interface the inversion assumes. The gap adds a round-trip linear phase
  that, mis-read as material, makes n fall and cross 1.
- **Approach.** Pure synthetic: plant a Drude CNT + known gap, forward-model the etalon, then
  apply the exact de-embed `x = (r_meas − r_front)/(1 − r_front·r_meas) = r_back·e^{−i2β}`,
  strip the 2β phase, re-invert.
- **Result.** `|x| = |r_back|` to 1e-16; n,k recovered to 4e-15 **when d is known**; the naive
  (no de-embed) n drops to 0.76 by 3 THz — reproducing the symptom. Estimating d from the
  slope of `arg(x)` is **biased** by the sample's own dispersion (34 vs 30 µm). Surface
  roughness (a spread σ_d) damps the high-frequency recovery (a coherence ceiling).
- **Conclusion.** The air-gap hypothesis is self-consistent and the de-embed is exact **if d
  is known**. The crux becomes: **how do we get a trustworthy d** (not the biased arg(x)
  slope), and roughness sets a fundamental high-f ceiling.

## 4. `deembed_air_gap_realdata` — does the de-embed lift n on REAL data?

- **Problem.** Confirm on measured CNT data that removing the gap lifts n above 1.
- **Hypothesis.** If the gap is the cause, de-embedding the real reflection should restore n≥1.
- **Approach.** Pull `r_meas` (=`reflection_r`) and `r_front` (=`r_reference`=r_{SiO2→air}=0.440)
  straight from the thz_core pipeline output; de-embed; estimate d from arg(x) slope and also
  sweep d; re-invert with air incidence at θ_gap=45°.
- **Result.** n **lifts from ~0.45 to above 1**. With the arg(x)-slope d (28 µm) it over-lifts
  to n≈3–4; the band first crosses n≥1 at d≈14–15 µm. n is *very* sensitive to d.
- **Conclusion.** The air gap **was** the cause — confirmed on real data. But the absolute n
  is only as good as d, and the arg(x)-slope d is the biased one. Need a robust d.

## 5. `deembed_air_gap_iterative` — how to pin d (iterative vs causality)?

- **Problem.** Get a trustworthy gap thickness d; the arg(x) slope over-lifts n.
- **Hypothesis.** Either iterate (assume smooth material, refit d) or use a causality
  (minimum-phase) constraint to separate the gap's linear phase from the material phase.
- **Approach.** Validate three d-estimators against **synthetic ground truth** (planted Drude
  CNT + known d = 20 µm) before touching real data: arg(x) slope, iterative-smooth, and
  minimum-phase (Hilbert of ln|x|).
- **Result.** Iterative-smooth is **degenerate** — it stays at the biased start (24.7 µm vs
  true 20). Minimum-phase **recovers d = 20.03 µm** and the true n exactly. On real data,
  minimum-phase gives d≈26–30 µm and lifts n to physical values, BUT the real excess phase is
  not perfectly linear (assumption imperfect) and the two orientations disagree by ~4 µm.
- **Conclusion.** The literal "iterate on the phase slope" idea **cannot** break the
  gap↔material degeneracy (the failure is fundamental, not numerical). Minimum-phase
  (causality) can — on a *smooth* gap. On the real *rough* contact it is only approximate.
  → the roughness itself must be modelled, or d taken from an assumption-light source (the
  pulse round-trip delay).

## 6. `air_gap_models` — is the inverse problem well-posed? (conditioning report)

- **Problem.** Before building full fitters for the roughness models, is the inverse problem
  well-posed for *our* sample, and exactly where does it become degenerate?
- **Hypothesis.** With a constrained material model (Drude–Smith) the gap and material are
  separable in principle, but weakly per single measurement; a pressure series should break
  it; and Kramers–Kronig/min-phase should fail once roughness corrupts the amplitude.
- **Approach.** Synthetic Drude–Smith CNT (n≈2.3, k≈1.2, |r_back|≈0.61) + a rough gap (d̄=17,
  σ_d=8 µm). Numerical Jacobian/Fisher analysis of the parameter uncertainties and the
  gap↔material correlations; single measurement vs a 4-run shared-material series; and a
  direct check of how fast roughness drives |x| away from |r_back|.
- **Result.**
  - Gap is well determined single-shot (d̄ ~1.2%, σ_d ~1.6%); material moderate (eps_inf 7.5%,
    ω_p 8.3%, τ 10.5%, c 6.7% at an assumed 2e-3 noise) — but **d̄ ↔ eps_inf correlation =
    +0.94**: the material is **degeneracy-limited**, not truly pinned, from one trace.
  - Useful lever: **material parameters speak at low frequency, the gap/roughness at high
    frequency** — a partial natural separation.
  - **Pressure series (4 runs) improves the material uncertainties ~2.2×** — it breaks the
    degeneracy, as hoped.
  - **KK/min-phase premise quantitatively fails under roughness:** |x|/|r_back| deviates 2% at
    σ=3 µm → 9% at 6 µm (marginal) → 19% at 10 µm (broken). At our ~8 µm that's ~10–15%
    magnitude corruption — so KK alone gives a biased material phase here.
- **Conclusion / recommendation.** The problem is **solvable** and the gap is well-constrained;
  the material is recoverable **via the pressure series**, not from single traces. KK is a
  *component* (de-corrupt |x| with the roughness model → KK phase), not a standalone fix.
  **Green-light** to build the two roughness fitters (statistical-gap FP, graded EMT), each
  synthetic-validated first, with the **pressure-series global fit as the actual deliverable.**
  Caveats: the uncertainty %s scale with the assumed noise floor (swap in measured SNR), and
  the analysis assumes the Drude–Smith model is correct (real model mismatch adds bias).

## 7. `compare_gap_models` — build, validate & compare both roughness models

- **Problem.** Do the two roughness pictures (statistical-gap FP vs graded EMT) actually fit
  the data, and do they agree on the recovered n, k, σ?
- **Hypothesis.** Both, with a constrained Drude–Smith material, should fit; comparing their
  residuals and recovered constants tells us which picture the data prefers (and whether the
  answer is model-robust).
- **Approach.** Implemented Route A (`rough_gap_reflection`) and Route B
  (`graded_emt_reflection`, Bruggeman, continuity-tracked root). Synthetic-validated each
  (plant→recover, 2e-3 noise) BEFORE real data, then least-squares fit both to the real
  CNT-0/90 `r_meas` from the pipeline.
- **Result.**
  - **Synthetic: both pass.** Route A recovers planted params to the conditioning-predicted
    ~10% (eps_inf), Route B to <1%; both hit the 2e-3 noise floor. The machinery is correct.
  - **Real data: both lift n above 1 and roughly agree on n,k at low f**, diverging on σ.
    Route A fits marginally better (RMS 6.2 vs 6.9e-2 for CNT-0; 5.4 vs 5.8e-2 for CNT-90).
  - **But neither fits well:** RMS ≈ 6e-2 is ~30× the synthetic noise floor. The measured |r|
    has structure (a dip near ~1.5 THz) that the smooth Drude–Smith + gap models do **not**
    capture.
  - **Material params are unstable** — Route A rails (τ→300 fs, c→−1, σ_d→0.1 µm: it wants a
    *sharp* single gap, not a rough one); Route B's eps_inf scatters unphysically between
    orientations (3.4 vs 19.7). Gap scales are plausible (A: d̄≈12 µm; B: z0≈25 µm offset,
    σ_h≈3–4 µm).
- **Conclusion / recommendation.** The models are sound (synthetic-validated) but a **single
  measurement cannot pin the material** — exactly the degeneracy the conditioning report
  predicted, now compounded by **model mismatch** (the |r| structure the smooth DS model
  misses). Two levers follow: (1) the **pressure series** (shared material across runs) to
  break the degeneracy — the planned deliverable; and (2) likely a **richer material model**
  (Drude–Lorentz / a resonance), since the measured |r| feature isn't reproducible by a plain
  Drude–Smith. Route A is marginally preferred on fit, and its σ_d→0 pull is itself a hint
  that a *bulk* gap (waviness) may dominate over micro-roughness for these samples — to be
  tested with the series.

---

## Current status & next steps

- **Established:** the sub-1 n is a real contact **air-gap Fabry–Pérot** effect (not
  processing); the exact de-embed works given d; the gap is **rough** (a distribution), which
  defeats both the phase-slope iteration and pure KK.
- **Decided:** model the roughness (statistical-gap FP and/or graded effective medium) with a
  constrained Drude–Smith material; use an **unquantified pressure series** fit to break the
  gap↔material degeneracy statistically. (Transmission is impossible for these thick opaque
  bundles.)
- **Both models built, validated, compared** (`air_gap_models.py` + `compare_gap_models.py`):
  correct on synthetic, but single-measurement real fits are degeneracy- AND mismatch-limited
  (RMS ~30× noise; |r| structure unmodelled; material params rail/scatter). n lifts above 1
  either way.
- **Next:**
  1. `fit_pressure_series.py` — global fit, **shared material across runs** + per-run gap; the
     trustworthy path. Synthetic demo now; real once the series is acquired.
  2. Try a **richer material model** (Drude–Lorentz / add a resonance) to chase the ~1.5 THz
     |r| feature the plain Drude–Smith misses — check if RMS drops and params stabilise.
  3. Independent **pulse-delay d** as a hard anchor on the gap (cross-correlate 2nd reflections)
     to relieve the gap↔material trade-off.
