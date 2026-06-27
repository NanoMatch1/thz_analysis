# Air-gap de-embedding — literature recanvas & recommended approach

**Question:** how to robustly recover CNT bulk optical constants (n, k, σ) from a window-coupled
reflection measurement with a thin, rough **contact air gap** — especially the **low-frequency**
band (carrier dynamics) — analytically. Recanvassed 2026-06-27; verifying whether minimum-phase
de-embed is the right approach or there is something better. Pairs with the lab notebook (F8, F9,
F11, F13, F15) and `reports/reflection_interface_theory_tutorial.md`.

---

## TL;DR recommendation

1. **The geometry is 80% of the answer, and the literature is unanimous: high-index coupling on
   HR-Si (ATR / Si-window reflection), sample in intimate contact (ideally deposited).** This is the
   *field-standard* way to do conductive-film THz, precisely because it fixes conditioning *and*, if
   the sample is deposited/grown on the prism, removes the gap. Put the build effort here (matches F13).
2. **For the residual gap, minimum-phase is the RIGHT FAMILY (causality-based phase retrieval) but the
   wrong *implementation*.** Two concrete upgrades the literature points to:
   - **Pin the gap's linear (timing) phase from the AMPLITUDE, not the measured phase.** The measured
     reflection phase is the unreliable quantity; |r| is robust. The arXiv "robust phase correction"
     paper (below) finds the misplacement/gap shift by **minimising the difference between the measured
     |r| and the |r| implied by a trial shift via Kramers-Kronig** — material-agnostic, no anchor, and
     *explicitly designed for highly reflective / conductive samples*. This is strictly better than our
     `detrend_transfer_phase`, which fits the (degenerate) measured-phase slope.
   - **Use Maximum-Entropy Method (MEM) instead of the Hilbert transform for the causal phase.** MEM is
     functionally equivalent to time-domain KK / minimum-phase but **handles finite bandwidth and the
     spectrum *edges* far better — and is specifically recommended for conductors.** The Hilbert
     min-phase we built has exactly the edge-artifact weakness we flagged, at exactly the low-frequency
     edge we need.
3. **Anchor with Si.** Both subtractive-KK and MEM improve markedly with 1–2 **anchor points** (a
   frequency where n is known). Our HR-Si calibration *is* that anchor.
4. **Model the gap's amplitude effect with the statistical-gap Fabry-Pérot (Route A)** only as a
   residual; with good contact it is small. Roughness enters as a Debye-Waller magnitude roll-off.

So: **min-phase was the right instinct, but the production tool is "MEM phase retrieval + amplitude-KK
gap-shift finder + Si anchor," on a Si-coupled measurement.**

---

## The literature, by stream

### A. Geometry: high-index coupling / ATR (the primary fix)
THz-TD-**ATR** with a **high-resistivity Si prism** is the established geometry for conductive/absorbing
samples: the sample sits on the prism base, the beam reflects (or totally-internally-reflects) at the
prism–sample interface, and the evanescent/Fresnel field probes the sample. HR-Si is chosen because it
is "almost perfectly transparent, non-dispersive and high-index in the THz band." Coupling through the
high index pulls the reflection off the `r=−1` pole (our conditioning argument, F7) **and**, when the
film is *deposited* on the prism, there is **no air gap**. This is the convergence of F7+F13.
- ATR-THz overview: [Frontiers, TD-ATR microfluidic](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1143443/full)
- CNT THz conductivity (deposited films): [SWCNT TDS, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0008622311008116),
  [anisotropic SWCNT conductivity](https://www.researchgate.net/publication/242215055_Terahertz_conductivity_of_anisotropic_single_walled_carbon_nanotube_films)

### B. Sidestep the phase entirely (amplitude-only)
For **thin** conducting films, a **Gires–Tournois étalon** (partial front reflector + conductor behind)
allows **amplitude-only** extraction of the sheet conductance — no phase needed, hence gap-robust.
- [Terahertz Characterization of Thin Conducting Films in Reflection (Gires–Tournois)](https://www.researchgate.net/publication/385188655_Terahertz_Time-Domain_Characterization_of_Thin_Conducting_Films_in_Reflection_Mode)
- **Caveat for us:** this is a *thin-film sheet-conductance* model; our CNT buckypaper is thick/opaque
  (semi-infinite to THz), so the bulk n,k — not a sheet σ — is what reflection sees. The *idea*
  (amplitude carries the conductance, phase is optional) is valuable, but the thin-film model doesn't
  transfer directly. Worth revisiting if we ever go to genuinely thin CNT layers.

### C. Phase retrieval from amplitude (causality) — the core de-embed tool
Because the measured reflection phase is corrupted (gap timing + alignment), reconstruct it from the
robust |r| via causality. Family members, weakest→strongest for our case:
- **Minimum phase via Hilbert(ln|r|)** — what we built. Correct in principle; **edge artifacts** on
  finite bandwidth (our low-f weak spot).
- **Singly/multiply-subtractive Kramers-Kronig (SSKK/MSKK)** — KK anchored at known points; better at
  the edges than plain KK.
- **Maximum-Entropy Method (MEM)** — functionally equivalent to time-domain KK but **best near the
  spectrum endpoints and specifically recommended for conductor materials**; benefits from **anchor
  points** (two > one).
  - [MEM vs subtractive-KK comparison (Lucarini et al., Appl. Opt.)](https://opg.optica.org/ao/abstract.cfm?uri=ao-45-25-6519)
  - [KK + time-domain combination for THz optical constants](https://www.sciencedirect.com/science/article/abs/pii/S0924203112000409)

### D. The robust timing/gap-shift method — the *worthy attempt* to adapt
**"Robust phase correction techniques for terahertz time-domain reflection spectroscopy"**
([arXiv 2412.18662v2](https://arxiv.org/html/2412.18662v2)). Corrects sample↔reference **axial
misplacement** (a uniform gap / linear phase `(ω/c)(2l/cosθ)`), which "is extremely sensitive in
reflection geometry." Two methods, both **inverse-KK based**:
- *Analytical fit* of the misplacement `l` from a derived expression for the phase error.
- *Amplitude minimisation* — compute |r| for trial shifts and pick the `l` that best matches the
  measured |r| (KK ties the two): `min_l ∫ ||r_calc(l)| − |r_meas|| dω`.
Material-agnostic, **no anchor points needed**, and **explicitly demonstrated on highly reflective /
conductive samples** (InSb plasma edge). **Limitation:** it treats the gap as a *uniform* axial offset
in air — **no rough-gap distribution and no two-interface FP**. So it solves our *linear-phase/timing*
part robustly; we bolt the **statistical-gap FP (Route A)** on top for the gap's amplitude (Debye-Waller
+ fringes), which only matters for a non-negligible/rough gap.

---

## Verdict on our current min-phase de-embed
- **Right family** (causality phase retrieval separating material phase from the gap's linear phase) —
  keep the idea.
- **Replace the engine:** Hilbert(ln|r|) → **MEM** (edge/conductor robustness at low f), with the HR-Si
  measurement as **anchor**.
- **Replace the gap-shift estimator:** measured-phase-slope fit → **amplitude-KK minimisation** (arXiv
  method) — robust and degeneracy-free, because it uses |r| not the corrupted phase.
- **Keep** the statistical-gap FP (Route A) for the residual amplitude correction; with good contact it
  is a small perturbation.
- **Demote** the time-domain-delay / `detrend_transfer_phase` to a *diagnostic*, not the correction.

---

## Recommended de-embed pipeline (target)
1. **Measure in Si-coupled reflection**, sample deposited/intimately pressed → conditioning fixed, gap
   minimised. (Geometry does most of the work.)
2. **Anchor:** HR-Si calibration run → known-n anchor point(s) for the phase retrieval + geometry
   (θ, n_Si) calibration.
3. **Gap-shift:** find the residual linear (gap) phase by amplitude-KK minimisation (arXiv 2412.18662).
4. **Phase retrieval:** reconstruct the material reflection phase from |r| via **MEM** (anchored).
5. **Residual gap amplitude:** if non-negligible, fit the statistical-gap FP (d̄, σ_d) — Debye-Waller +
   fringes — jointly with a **constrained Drude/Drude-Smith** material model (the material's curved
   dispersion breaks the linear-phase degeneracy; F11).
6. Invert → n, k, σ; report only where conditioning + anchoring support it.

## What to implement next (code)
- `mem_phase_retrieval(|r|, anchor)` — replace `minimum_phase_from_magnitude` (Hilbert) with MEM.
- `gap_shift_amplitude_kk(r_meas)` — the arXiv amplitude-minimisation shift finder.
- Wire both into the de-embed step; keep Route A FP as the residual-amplitude model.

## Open questions / for further reading (Samuel + me)
- `RQ1` Does MEM's edge advantage actually recover usable n below ~0.7 THz for *our* SNR? (test on Si first.)
- `RQ2` Best ATR vs near-normal-incidence Si-window trade for a *pressed* (not deposited) CNT paper —
  is TIR even reachable/useful given CNT's complex high index?
- `RQ3` Any THz paper handling a *rough/distributed* contact gap explicitly (beyond uniform offset)?
  (none found yet — this may be our contribution.)
- `RQ4` Tinkham/sheet-conductance only if we can make genuinely thin CNT layers on Si.

*Sources consolidated above as inline links; this supersedes the scope in the older
`reports/cnt_rough_gap_literature_canvas.md` for the de-embed question specifically.*
