# Implementation plan — statistical-gap & effective-medium models for rough-contact CNT reflection

Date 2026-06-22. Scopes two analytic-modelling routes to recover CNT optical constants
through a *rough contact gap*, run as isolated test scripts on the 2026_06_19 data, with
the thz_core reflection pipeline as the front-end machinery (it supplies the measured
reflection) and the models implemented ad hoc in `explorations/`.

Constraints from Samuel (2026-06-22): transmission is impossible (thick, opaque, highly
conductive macroscale CNT bundles — already tried); pressure can't be quantified, so a
pressure *series* fit to statistical trends is the realistic lever.

## 0. Grounding & key assumptions (validated against the literature)

- Reflection mode is mandatory (opaque conductor). Roughness *enhances* attenuation with
  frequency (the coherence/Debye–Waller rolloff) — matches our roughness demo and the
  metallic-roughness THz literature (Kirchhoff/Beckmann–Kirchhoff).
- Regime check: gap & roughness ~10–25 µm ≪ λ_THz (300 µm–1.5 mm over 0.2–2.5 THz). So we
  are in the *coherent thin-film + quasi-static effective-medium* regime — both models are
  valid; we are NOT in geometric-optics scattering.
- Material model for CNT: Drude / Drude–Smith / Drude–Lorentz (+ Maxwell–Garnett for
  aligned films); anisotropic (longitudinal vs transverse) — consistent with CNT-0 vs
  CNT-90. The *constrained* material dispersion is what breaks the gap-vs-material phase
  degeneracy that defeated the naive phase-slope iteration.
- Sensitivity sanity: the de-embedded CNT gives n,k ~ 3–5, so r_back = r_{air→CNT} ≈ 0.7 in
  magnitude — NOT a near-perfect mirror, so reflection retains usable sensitivity to n,k.
  (We still quantify the Jacobian conditioning in the synthetic test — see §6.)

## 1. Shared infrastructure (both routes)

`explorations/air_gap_models.py` (ad hoc module, not pipeline):
- `load_measured_reflection()` — runs `run_shared_axis` (from compare_reflection_pathways)
  and returns per sample: freq, mask, r_meas (=reflection_r), r_front (=r_reference),
  naive n,k. The pipeline is the core machinery; the models below consume its output.
- Material models: `epsilon_drude_smith(omega; eps_inf, omega_p, gamma, c)` and a
  `epsilon_drude_lorentz(...)` variant; `reflection_air_to_cnt(omega, eps, theta_gap)` via
  `core.fresnel_reflection_s`. Anisotropy = fit each orientation's own params.
- Fitting: scipy `least_squares` on the complex residual Re/Im of (r_model − r_meas) over
  the trusted band, with bounds. (thz_core fitting registry deferred — ad hoc for now.)

## 2. Route A — statistical-gap Fabry–Pérot + roughness

Geometry: SiO2 | air gap d | CNT (semi-infinite). Single-gap reflection (exact):
    r(d,ω) = (r_f + r_b(ω) e^{−i2β}) / (1 + r_f r_b(ω) e^{−i2β}),   2β = (ω/c)·2d·cosθ_gap
with r_f = r_{SiO2→air} (known, 0.440) and r_b(ω) = r_{air→CNT}(ε_CNT(ω)).

Rough contact = a distribution P(d). Measured = beam-spot average:
    r_meas(ω) = ∫ r(d,ω) P(d) dd
- A1 (default, exact): numerical average over a fine d-grid / Gauss–Hermite nodes — keeps
  all bounces. P(d) = Gaussian(d̄, σ_d), truncated at d ≥ 0.
- A2 (intuition/seed): single-bounce closed form r_meas ≈ r_f + (1−r_f²) r_b e^{−i2β(d̄)}·W(ω),
  with the Debye–Waller factor W(ω) = exp[−2((ω/c)cosθ_gap σ_d)²] — the Rayleigh roughness
  factor. Useful seed; the |x| rolloff in our roughness demo IS this W.
Fit params: {ε_CNT params}, d̄, σ_d.  Refinement to flag: a *contact fraction* (delta at
d=0) + Gaussian tail, for partially-contacting rough surfaces.

## 3. Route B — graded effective-medium (EMT) layer

Geometry: SiO2 | graded air→CNT layer (thickness L, N sublayers) | bulk CNT. No discrete
gap; a density gradient (the micro-roughness picture).
- Fill fraction vs depth from the height distribution: f(z) = Φ((z−z0)/σ_h), the Gaussian
  CDF (Aspnes-style EMA roughness layer). z0 = mean surface position (a pure-air top
  sublayer if z0 ≫ σ_h ⇒ also captures a bulk gap).
- Per-sublayer effective permittivity by Bruggeman (handles percolation of the conductive
  network): f·(ε_CNT−ε_eff)/(ε_CNT+2ε_eff) + (1−f)·(ε_air−ε_eff)/(ε_air+2ε_eff) = 0.
- Reflection by transfer matrix / recursive Airy through the N sublayers, s-pol, incident
  from SiO2 at θ_sio2.
Fit params: {ε_CNT params}, σ_h, z0 (and L, N fixed/large enough).
Picture vs A: A = flat CNT at a varying *distance* (waviness ⇒ gap distribution); B =
micro-rough surface ⇒ density *gradient*. Reality is a mix; comparing residuals tests which
dominates.

## 4. The degeneracy-breaker and the pressure-series design

Single measurement: even with a material model, gap and material are only *weakly*
separable (their ω-signatures differ — Drude rolloff vs Debye–Waller Gaussian + linear
phase — but not orthogonally). Treat single-measurement fits as exploratory.

Pressure series (the real lever): N measurements at unknown, varying pressure. Global fit
with SHARED material params θ (same CNT) + PER-measurement gap params (d̄_i,σ_d,i [A] or
σ_h,i,z0,i [B]). The shared θ across many runs massively over-determines the material →
pins it even though each run is degenerate. "Zero-gap extrapolation" = the converged shared
θ. Optionally impose monotonic gap-vs-run ordering if the pressure rank is known. This is
exactly Samuel's "series of measurements, fit trends statistically."

## 5. Validation strategy (synthetic-first — non-negotiable)

For each route: plant known {ε_CNT (Drude–Smith), gap params} → forward-model r_meas → fit
→ recover? Specifically:
- Single-measurement recovery + a Jacobian/Fisher conditioning report (how well-determined
  is each param; quantify the gap↔material degeneracy).
- Synthetic *series* (shared material, several gaps) → does the global fit recover the
  material when a single run cannot? This proves the series breaks the degeneracy BEFORE we
  trust real data, and tells us how many pressures we need.

## 6. Risks & caveats (explicit)

1. **Degeneracy** (gap ↔ material) for a single measurement — the central risk; the
   pressure series is the cure; quantify it in §5.
2. **Conditioning** — appears OK (|r_back|~0.7) but verify via the synthetic Jacobian; if a
   future sample is a near-perfect mirror, n,k become intrinsically hard.
3. **Material-model mismatch** — if CNT σ(ω) isn't Drude–Smith, the fit biases; check
   residuals, try Drude–Lorentz / Maxwell–Garnett.
4. **A vs B ambiguity** — both may fit; discriminate by residuals + whether recovered gap
   params (d̄~10–25 µm, σ_h) are physically sane.
5. **P(d) truncation / contact fraction** — start Gaussian; add contact-delta if needed.
6. **EMT mixing rule** — Bruggeman (percolation) over Maxwell–Garnett for a conductive net.
7. Inherits the pipeline's phase referencing (r_meas, r_front) — already validated.

## 7. Deliverables & sequencing

Scripts (all in `explorations/`, all reuse `run_shared_axis` for the measured reflection):
1. `air_gap_models.py` — shared module (material models, FP+average, EMT+transfer matrix,
   fitting, pipeline loader) + a **conditioning/sensitivity report** on synthetic data.
   *Do this first* — it answers "is the inverse problem well-posed for our sample?".
2. `fit_statistical_gap.py` — Route A: synthetic validation → single real fit → diagnostics.
3. `fit_graded_emt.py` — Route B: same structure.
4. `compare_gap_models.py` — A vs B on the real CNT-0/CNT-90 data (residuals, recovered n,k).
5. `fit_pressure_series.py` — global shared-material fit; synthetic demo now, real when the
   pressure series is acquired.

No pipeline edits; models stay ad hoc in explorations until a route proves out.

## Sources (this round + the canvas)

- Roughness parameters of metallic surfaces from THz reflection (Opt. Lett.): https://opg.optica.org/ol/abstract.cfm?uri=ol-34-13-1927
- Influence of surface roughness on conductor at THz (Optik/ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S0030402614001478
- Effects of surface roughness on THz spectra (Springer): https://link.springer.com/content/pdf/10.1007/s11082-020-02365-x.pdf
- Modified Beckmann–Kirchhoff for slightly rough surfaces at THz (IEEE): https://ieeexplore.ieee.org/document/8889095
- Bruggeman EMA for porous/rough layers (porous silicon): https://www.researchgate.net/publication/229914097
- Heterogeneous dielectric mixtures in the THz regime — quasi-static EMT: https://www.researchgate.net/publication/231015899
- Effective media with gradient dielectric function for rough surfaces (arXiv): https://arxiv.org/pdf/1907.03057
- Super-aligned MWCNT THz dielectric (Drude–Lorentz + Maxwell–Garnett, anisotropic): https://www.nature.com/articles/s41598-018-20118-5
- THz conductivity of anisotropic SWCNT films: https://www.researchgate.net/publication/242215055
- THz time-domain characterization of thin conducting films in reflection: https://www.researchgate.net/publication/385188655
- (Plus the full canvas: `reports/cnt_rough_gap_literature_canvas.md`.)
