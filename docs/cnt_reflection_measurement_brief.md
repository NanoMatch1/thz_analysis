# Measuring highly reflective (CNT) samples in THz reflection — technical brief

> **What this document is.** A single, self-contained orientation on *why* measuring
> highly reflective, metallic-like samples (carbon-nanotube "buckypaper") by THz-TDS
> reflection is hard, what physics makes it hard, every geometry and trick we have tried,
> and where the effort now stands. It is written so an AI agent (or a person) can read it
> once and be up to speed enough to evaluate a new technical idea against what we already
> know — without first reconstructing the story from the scattered primary sources.
>
> **This is a synthesis, not the source of record.** The dated, never-erased findings log
> is `reports/CNT_measurement_lab_notebook.md` (findings `F1`–`F29`); the physics-decision
> log is `ANALYSIS_NOTES.md` (sections `§N`); the pipeline data-flow is
> `PIPELINE_ARCHITECTURE.md`. This brief cites both by `F##` and `§N` so you can drill in.
> If this doc and the lab notebook disagree, **the lab notebook wins** (it is maintained
> finding-by-finding; this is refreshed periodically).
>
> Sign/units convention (matches the codebase): `n̂ = n − ik`, `k ≥ 0` absorption,
> time convention `exp(−iωt)`. `thz_core` works in SI (s, Hz); the adapter converts to
> display units (ps, THz) only at the boundaries.

---

## 0. TL;DR — the one-paragraph version

CNT buckypaper is a free-carrier (Drude) conductor, so it reflects THz almost like a
mirror: its reflection coefficient sits near `r = −1`, exactly where the closed-form
`r → n̂` inversion has a **pole** and is catastrophically ill-conditioned — worst at low
frequency, which is precisely the band where the carrier dynamics we want to measure live.
Coupling the beam through a **high-index window** (SiO₂ ≈ 1.95, ideally HR-Si ≈ 3.4) is an
ATR/immersion trick that moves `r` off the `−1` pole and reopens the low band — *the*
enabling move. But pressing paper against a window leaves a **µm-scale air gap**; the wave
then reflects off *air*|CNT again (throwing away the conditioning advantage) *and* the gap
adds an opposite-sign SiO₂→air etalon reflection that drags the inverted `n` below 1. The
gap is **time-invisible** (a 4 µm / 19 fs gap is below pulse-centering resolution yet wrecks
low-f `n`) because near-mirror conditioning amplifies any phase residual ~10×. We built:
front-pulse **self-referencing** (cancels window/mount drift structurally), a validated
Fabry–Pérot **air-gap de-embed** with min-phase/MEM gap estimation, and **p-polarization**
inversion + dual-pol joint fitting (p-pol suppresses the parasitic front reflection *and*
adds ~6× pole-distance). Verdict as of F28: the *analysis* is at its information limit —
Si controls confirm the pipeline is correct (n = 3.418 in both pols) and the instrument is
at the ~0.5–2 % level; the binding wall is now **sample presentation** (a smooth ~10 %
mount-coupling systematic + air-incidence conditioning behind any residual gap), not code.
The path forward is CNT deposited/pressed on rigid flat **HR-Si**, measured *through* the Si.

---

## 1. The sample and the science goal

- **Sample:** carbon-nanotube "buckypaper" — a pressed mat of nanotubes. Electrically a
  **free-carrier (Drude) conductor**. Often **anisotropic** (aligned fibers → σ differs
  along vs across the fiber axis; we see it in the 0°/90° mount rotations, F23/F28).
- **Goal:** extract complex optical constants `n̂(ω) = n − ik` and optical conductivity
  `σ(ω)` — specifically the **carrier dynamics**: the Drude peak shape and scattering time
  `τ` (`1/τ ~ 1 THz`), which live **at and below 1–2 THz** (F15). This is why "just work at
  high frequency where conditioning is easier" is not an option — the diagnostic band is the
  hard band.
- **Instrument:** THz-TDS with a spintronic emitter (polarization is a *free knob* — rotate
  the magnet, rotate the EO detection crystal to match; this is what makes the p-pol route
  cheap, F21) and a GaP electro-optic detection crystal (source of the +5.3 ps instrumental
  "echo" and a second-order ~168 ps echo we must gate out — `§2`, `§13a`).

---

## 2. Why it is hard — the core physics chain

The chain you invert (tutorial: `reports/reflection_interface_theory_tutorial.md`):

```
free carriers → ε(ω) → n̂ = √ε → r(ω ; N₁) Fresnel → measure → invert r → n̂ → σ
     Drude       permittivity    complex index    reflection          (closed form)
```

Four compounding problems (lab notebook F1–F13):

### 2.1 Near-mirror conditioning: the `r = −1` pole  [F1, F2]
The closed-form inversion is `q = N₁cosθ·(1−r)/(1+r)`, `n̂ = √(q² + (N₁sinθ)²)`. The
denominator `(1+r)` gives a **pole at r = −1**. A Drude conductor is most reflective at DC
(the Drude peak is at ω = 0), so `r → −1` **worst at low frequency** — least contrast, least
information, exactly where we need it. Distance to the pole is `|1 + r| = |1 − H|`; a small
error `δr` gives `δn ∝ δr / |1+r|²`.

**Subtlety (F2):** the pole has a *neighborhood*. `|1−H|` depends on **both `|H|` and
`arg(H)`**, not magnitude alone — so even a sample with `|H| < 1` ("intensity down, no
crossing") spikes `n` wherever `arg(H)` rotates `H` real-positive. The `min_one_plus_r` mask
floor catches the pole itself but not its neighborhood. This is the mechanism behind the
spurious "~0.6 THz Lorentzian" peak in CNT `n` — it is an **inversion artifact at a k-dip**,
moves with the de-embed gap, and vanishes at the physical gap (F10, memory
`cnt-lorentzian-is-inversion-artifact`).

### 2.2 Alignment / wavefront on bare CNT  [F3, F4, F5, F6]
The CNT mat cannot be aligned with the visible generation laser (it does not back-reflect
the NIR alignment beam), so the sample-face plane is uncontrolled run-to-run ("flatten,
place, and hope", F3). The paper's roughness is **large-scale undulation** (correlation
length ≫ λ), not within-spot RMS (F4): sub-wavelength micro-relief is Debye-Waller-invisible
to 300 µm THz, but the undulation **tilts the wavefront** → beam-steering. That distorts the
*pulse shape* asymmetrically, which injects a **group delay into `arg(H)` even when the peaks
are perfectly time-aligned** (F5) — and that group delay sweeps `arg(H)` through zero
periodically, triggering the F2 pole-neighborhood spikes. Peak-recentering cannot fix it
(peak position ≠ group delay for a distorted pulse). `|H| > 1` is the actionable alignment
alarm here — a passive sample cannot out-reflect gold, so `|H|>1` means the reference
coupled worse than the sample (F6). *(Caveat: `|H|>1` is retired as an alarm in the **window**
geometry — see `§5.4`.)*

### 2.3 The bare-reflection geometry is *structurally* wrong for the goal  [F1, F15]
Bare air|CNT reflection is simple and contact-free but loses the low band to conditioning as
a matter of **fundamental physics**, not fixable processing. Good alignment fixes the
*wavefront* problem but **not** the conditioning problem. So bare reflection is kept only as a
**> 2 THz cross-check**, never the primary instrument (F15).

### 2.4 The pipeline itself is NOT the problem  [F12]
The transmission pipeline recovers a *known* `n` to `|Δn| ≈ 0.001` on public synthetic data
(phoeniks); the only deviation is mild under-resolution of *narrow* absorption lines (CNT has
none). Si in the exact reflection geometry gives `n = 3.418` in both polarizations
(F25, F26). ⇒ inter-measurement variation is **sample + geometry**, not code.

---

## 3. Geometries — used and considered

| Geometry | `geometry=` | Reference | Incident medium `N₁` | Status / use |
|----------|-------------|-----------|----------------------|--------------|
| **Window-coupled reflection** | `'window'` | computed `r_{SiO₂→air}` | SiO₂ ≈ 1.95 | **Primary.** CNT pressed on back face of SiO₂ window; 45° external → ≈21.3° internal (Snell). Two pulses per trace (front SiO₂ face + sample back face). |
| **Gold-referenced bare reflection** | `'gold'` | `r_gold ≈ −1` | air = 1 | Cross-check / Si benchmark. Single pulse, no window, no segmentation. Ill-conditioned for conductors. |
| **Transmission** | `invert_nk` (separate path) | open beam or substrate | air = 1 | Not viable for CNT (too conductive/thick, F11). Used for substrates and the Si σ benchmark. |
| **HR-Si window (proposed)** | `'window'`, `n_window≈3.4` | Si back-face self-ref | Si ≈ 3.4 | **The planned endgame** (F13, F16). Highest conditioning headroom; deposit/press CNT directly on flat HR-Si → alignment plane + conditioning + gap-free contact in one jig. |

**Geometry detail** lives in `ANALYSIS_NOTES §1` (angles, sign convention, computed
back-face reference). The **sign of the reflection is the discriminator, not the amplitude**:
a reflection off a *higher*-index medium inverts (π flip); SiO₂→CNT is negative (arg ≈ π),
SiO₂→air is positive. Overlaying sample vs bare-window second reflection: opposite polarity ⇒
seeing the high-index sample; same polarity ⇒ air gap (`§1`).

---

## 4. The high-index window — why it rescues conditioning  [F7]

Reflection is an **impedance mismatch**, `r ≈ (Z_sample − Z_incident)/(Z_sample + Z_incident)`
with `Z ∝ 1/N`. A conductor has very low impedance; seen from **air** (high impedance) the
mismatch is enormous → `r ≈ −1` (that is why metals are mirrors). **Raising the incident
index `N₁`** (couple through SiO₂ or Si) *lowers* the incident impedance toward the
conductor's → the mismatch shrinks → **`r` moves off the −1 pole** → more wave couples in →
the reflection carries more information about `N₂`. This is textbook **ATR / immersion /
prism-coupling** spectroscopy of strongly reflecting materials.

Near the pole, `|1+r| ≈ 2·N_incident / |N_sample|`, so a higher incident index keeps `r`
further from `−1` roughly by the factor `N_incident`. Measured pole-distance `|1+r|` at
0.3 THz for our Drude CNT: **air 0.06 → SiO₂ 0.12 → Si 0.20** (`§6b`, tutorial §5). Si sits
~3× further from the singularity than air, so a given measurement error does ~3× less damage
(amplitude; ~10× in variance). **It is the incident *medium* that matters, not the reference
choice** — swapping gold→SiO₂ as the reference while the CNT still sits in air does nothing.

Measured payoff (`§6b`): air/gold CNT reflection blows up (n→120, σ₂→−12500 S/cm at
1.5–1.7 THz where r≈−1); the SiO₂-window CNT is stable and bounded (n≈4.5 flat, k≈0.5–1). So
**the window is not a contact convenience — it fundamentally improves the inversion for
conductors.** (The gold/air path also mis-recovers Si n if T0 alignment is off — see `§7`.)

---

## 5. The catch: the air gap at the sample–window interface

This is the central technical obstacle of the window geometry.

### 5.1 It is a gap, not intrinsic reflection delay  [§9, §9b, F8]
A re-pressed CNT-on-glass showed a 0.155–0.18 ps T0 shift that changes with 90° rotation.
Intrinsic reflection group delay is tiny (a lossless interface is frequency-flat, a
π flip is a *sign* not a delay; a metal is ~skin-depth/c ≈ sub-fs; a Drude conductor at most
~τ, tens of fs, and frequency-*dependent*). A flat 0.18 ps ramp ⇒ a **~33–38 µm air gap**
(`Δt = 2·n_gap·d·cosθ_gap/c`). Audited on real CNT-17 data (`§9b`): `arg(H)` carries a
consistent **+0.107…0.127 ps linear phase** across all samples/rotations ⇒ a 16–19 µm
equivalent gap, systematic and instrumental — confirming the gap, not a code bug. The
reflection phase code is provably immune to the transmission-style unwrap bug (`§9b` — the
closed form works on complex `r`, an integer 2π is invisible to it).

### 5.2 Two mechanisms, both bad  [F8, F9]
1. **Interference contaminant (F8).** SiO₂|air|CNT gives **two** reflections: SiO₂→air
   (n₂>n₃ → *positive*, ≈+0.32) and a gap-delayed air→CNT (n₂<n₃ → *negative*, ≈−1). Perfect
   contact would give a *single* negative SiO₂→CNT reflection; the gap splits it into a
   positive-then-negative pair. The spurious **positive** SiO₂→air term is the contaminant —
   when gap roughness washes out the negative CNT term, the positive term dominates and
   **drags inverted n below 1**.
2. **Conditioning destroyed (F9).** Because the wave crosses the gap *as air*, the
   sample-carrying reflection happens at an **air/CNT** interface → you snap right back to the
   air-incidence singular case (`§2.1`) *and* add the interference. The gap is **doubly
   damaging**. Optical contact is non-negotiable.

### 5.3 The gap is time-invisible — the crucial insight  [F19]
A **4 µm gap is 19 fs — below the ~25 fs pulse-centering resolution, i.e. invisible in the
time trace — yet it shifts low-f CNT `n` by −0.84 at 0.5 THz.** To hold `Δn(CNT) < 0.05` needs
< 1.5 fs centering, beyond any pulse-centering method. **The time-domain peak is ~10× too
coarse a ruler to certify gap-free contact for a near-mirror sample; the frequency-domain
inversion is the sensitive instrument.** The same gap barely moves Si (13× less — Si is less
metallic, better conditioned). This resolves the **roughness paradox**: the bare-reflection
surface and the window-gap are the *same paper topography in two geometric roles* — bare =
micro-relief on a mirror (sub-λ, Debye-Waller ≈ 1, invisible); window = the *spacer of a
coherent etalon* (round-trip phase read directly, conditioning-amplified, catastrophic).
Roughness didn't change; its job did.

**Mean gap `d̄` vs spread `σ_d` (F17):** `d̄` is a deterministic frequency-dependent
interference present even for a perfectly *smooth* gap — this is the dominant,
de-embeddable enemy. `σ_d` (roughness spread) is a Debye-Waller fringe-contrast loss
`exp(−2(ω/c·σ_d cosθ)²)` that scales ∝ω² → bites at **high** f, negligible in the low band we
care about. **Kill the mean gap; don't over-invest in roughness statistics.**

### 5.4 De-embedding the gap — the math is done and validated  [§9, F11, F16, F18, F26]
The gap is a Fabry–Pérot layer:
`r_meas = (r1 + r2·e^{−2iβ})/(1 + r1·r2·e^{−2iβ})`, `r1 = r_{SiO₂→air}` (known),
`β = (ω/c)·d·cosθ_gap`. Solve exactly for the CNT term:
**`x = r2·e^{−2iβ} = (r_meas − r1)/(1 − r1·r_meas)`** (well-conditioned, denominator ≥ 0.56).
Then `|x| = |r2|` exactly, and `arg(x) = arg(r2) − 2β` where `2β` is a pure linear phase.

- **Recovering the gap `d` is the weak link, not the algebra.** With the *true* d, n,k
  recover to 4e-15 (`§9` prototype). But a naive linear fit of `arg(x)` **overestimates d**
  because the sample's own dispersion adds phase slope. So d must come from the pulse
  round-trip delay, or be estimated by **causality-based phase retrieval** (F16):
  - **Phase-excess estimator** (`estimate_gap_minimum_phase`): the production tool for the
    small good-contact gap — immune to the conductor sign, works sub-fringe (F18).
  - **MEM (maximum-entropy) phase retrieval** — ~6× better than Hilbert at the low band edge
    on band-limited data (our regime), conditional on a near-edge feature (F18, memory
    `mem-phase-retrieval-implemented`).
  - **Amplitude-KK** (arXiv 2412.18662 idea): validated but **fails for our small gap**
    (conductor sign ambiguity + sub-fringe degeneracy — a gap < ~30 µm is sub-fringe in our
    band). It earns its keep only for **super-fringe** gaps (≳ 60 µm) and as an
    alignment-immune cross-check (F18).
  - **HR-Si anchor** — a known-n control in the same slot calibrates the systematic bias
    (±1 sample of timing registration ≡ 2.65 µm of apparent gap; the Si control removes it,
    F26).
- **Validated on both Si controls (F26):** the full chain (x-extraction → phase-excess d →
  strip 2β → air-frame re-inversion) flattens Si-s to `median|n−3.418| = 0.058`; Si-p's
  algebraic x-extraction alone gives flat `n = 3.418 ± 0.04`. The **de-embed math is done**.
- **Pipeline step:** `deembed_air_gap_reflection` (opt-in, between invert and derive;
  `PIPELINE_ARCHITECTURE §6`). Interactive `air_gap_slider_explorer.py` for manual
  position/width tuning with live n/k/σ. **Bug history (F29):** it once hardcoded
  `polarization='s'` and stripped the front + re-inverted at air incidence even at zero gap
  (collapsing gapless samples to the grazing floor n≈0.707) — both fixed; zero gap is now a
  genuine no-op.

---

## 6. Self-referencing — the drift/window cancellation we developed  [§11, §14]

**Problem:** the bare-window reference and the pressed-sample measurement are *different
mounts*. Window deformation under pressure/rotation/realignment changes the coupling, so the
reference no longer matches the sample's window path → frequency-structured fake features in
H.

**Key identity (`§11`).** The intra-trace ratio `W(ω) = Y₂/Y₁` (back pulse / front pulse) of
a single trace is a property of the **window alone** — source spectrum, detector response and
the shared air path are common to both pulses and cancel. The front pulse (air→SiO₂ face)
**never sees the sample**. So each trace can be referenced to its own front pulse.

**Three transfer modes** (`transfer_function`, `PIPELINE_ARCHITECTURE §6`,
`ANALYSIS_NOTES §11/§19`):

```
self_reference : H = (Y2_s/Y1_s)/(Y2_r/Y1_r)        front pulse for TIMING + amplitude
self_phase     : H = (Y2_s/Y2_r)·(C/|C|), C=Y1_r/Y1_s   front pulse for TIMING only
plain          : H = Y2_s/Y2_r                       (needs align_to_reference for timing)
```

- **`self_reference`** trades a 7–10 % structured mount-drift systematic for a ~1.6 % noise
  floor; on CNT-13/D the repeat spread drops n 11%→4%, σ₁ 18%→5% (`§11`). Real anisotropy
  (90° rotation) survives cleanly.
- **`self_phase`** (newer, `§19`) uses the front pulse purely as a *timing* reference
  (unit-magnitude `C/|C|` cancels a rigid acquisition Δt structurally) **without** importing
  the front amplitude — a robust frequency-domain replacement for time-domain alignment.
- **`C = Y1_s/Y1_r` is a diagnostic, not a corrector** (`selfref_quality`, `§10`): `H/C`
  re-injects the timing that self-referencing cancels. `std(|C|)` separates good (~0.01) from
  bad (~0.53) mounts; `|C|` medians finger the mount coupling (CNT 0.85 vs Si ~1.0, F28).

**Abs-time phase reference (the load-bearing nuance, `§11/§14`, Audit 1/2, memory
`audit-1`/`audit-2`):** both reflection segments get `exp(−2πi·f·t[0])` w.r.t. their own
`t[0]` so `W = Y₂/Y₁` encodes *only* the physical inter-pulse delay and is shift-invariant.
Applying it to one segment only leaks a linear phase into H. This is why **no time-domain
alignment is used in the self-ref path** — `align_to_reference` was not just unnecessary but
*harmful* here (it shifted only the sample → leaked phase; on real Si it mis-recovers
n = 2.30 vs the true 3.40, while self-ref/self_phase both give 3.40 — `§19`).

**Window characterisation bonus (`§11`):** the same `W` inverts for `n_SiO₂(ω)`
(`characterise_window` / `thz_core.invert_window_index`) — CNT-13/D gives n = 1.962–1.967
flat ⇒ **fused silica** (crystal quartz would be ~2.1), so the rotation sensitivity is
geometric (window wedge / mount), not birefringence.

---

## 7. p-polarization and dual-pol ellipsometry — attacking gap + conditioning at once  [F21–F29]

Samuel's idea (F21): the spintronic emitter makes polarization a free knob, so run a
**polarization series** instead of an (impractical) pressure series. p-pol attacks **both**
core problems simultaneously:

1. **Front suppression.** The parasitic SiO₂→air front reflection drops `|r|` 0.44 (s) →
   0.19 (p) at 45°, → **0 at the Brewster external angle ~63°** → removes the gap
   interference (F8) *at its source*.
2. **Conditioning.** The air→CNT back `|r|` falls ~0.85 (s) → ~0.55 (p) → off the r=−1
   mirror. Measured pole-distance `|1+r|`: **s-pol 0.30–0.40 vs p-pol 1.81–1.90 — ~6×**
   further off the pole (F23), largely via the *sign* of the p-pol front reference
   (r_front: s = +0.44, p = −0.195).

**Implementation (F22, ported to pipeline):**
- **p-pol inversion** in `thz_core.invert.invert_nk_reflection` (`polarization='p'`) — a
  closed-form quadratic in N₂² with a genuine **root ambiguity** (physical index vs a "gain
  twin", both reproduce r_p to machine precision) resolved by a passivity prior. `run_me_low-level.py`
  handles a p-pol dataset by just setting `config['geometry']['polarization']='p'`.
- **Dual-pol joint fit** (`thz_core.reflection_gap.fit_reflection_gap_dual_pol`) — fits one
  n(ω) + one shared gap `d` to **s and p together**; recovers planted n+d exactly, and beats
  single-pol under noise. Driver `run_me_dual_pol.py`. KK phase correction is
  polarization-independent (only the downstream `r→n` inversion picks s vs p, `ANALYSIS_NOTES`
  KK section).

**Hard-won caveats:**
- **Self-referencing needs NO s/p channel calibration** (F21) — each pol self-references
  against its own front pulse, so the channel response cancels inside H. Only *bare*
  reflection would need channel calibration.
- **Naming (F24):** in `X-Y`, X = THz pol vs the plane of reflection (0 = S, 90 = P),
  **Y = sample/mount orientation** (not detection pol — detection rotates *with* the THz pol).
  The SiO₂ faces are non-parallel, so rotating the mount precesses the front reflection around
  the EO detection cone → **each sample must pair with the reference at its own pol+orientation
  slot**. F23's original pairing was wrong for 0-90/90-0.
- **Per-mount gaps break the "shared gap" assumption (F27).** To see the same fiber axis with
  both pols you must rotate the sample → re-mount → *new* gap (0-0: 3.4 µm vs 90-90: 5.3 µm).
  So the dual-pol joint fit's one-shared-gap assumption is structurally wrong for anisotropic
  samples; it needs per-channel gaps. This (not mispairing) is a core reason the joint fits rail.
- **p-pol trades the r=−1 pole for a root-swap boundary (F26, F29).** Small gap-phase errors
  flip the inversion branch. The old per-bin passivity picker chose the wrong root wherever the
  corrupted low-f measurement flipped `sign(Im r)` → n collapsed to the **grazing root
  `n₁ sinθ = 0.707`** below ~0.7–1 THz. Fixed (F29) with a **global branch vote**. For p-pol the
  gap must be pinned externally (Si anchor / s-channel), never trusted from a phase-slope fit alone.
- **`|H| > 1` is retired as an alarm in the window geometry (F28).** The window reference
  back-reflection is *weak* (s: +0.44, p: −0.196), so any sample reflecting more gives `|H|>1`
  **physically** (p-pol Si: expected H = −1.27, measured |H| = 1.271!). The 76–100 % `|H|>1`
  fractions read as "alignment damage" in F20/F23 were largely expected physics. `|H|>1` is only
  meaningful where the expected `|H| < 1` (e.g. s-pol Si, gold-referenced bare geometry).

---

## 8. Where it stands — the verdict  [F28]

The error budget is closed (F28):

- **The measurement is excellent.** Independent realignment repeats agree to **0.6 % in |H|,
  1.5 fs timing, 0.003 rad detrended phase**; reference slot-to-slot 2 %. The old "alignment
  is the limiter" reading (F20/F23) is **overturned** for this dataset.
- **The pipeline is validated.** ±0.001 on public transmission data (F12); Si at 3.418 in
  **both** polarizations in this exact reflection geometry (F25/F26).
- **The gap is measured and de-embeddable.** ~5.3 µm raw phase-excess, ~1–3 µm after the
  Si-anchor systematic correction (F27) — good contact, right at the F19 4 µm sensitivity
  threshold. De-embed math validated on both Si controls (F26).
- **But every material+gap model leaves a smooth ~9–13 % residual** (plain Drude,
  Drude-Smith, free/fixed/Si-anchored gap, per-channel nuisance scale+delay — all rms
  0.08–0.13 vs the 0.005 reproducibility floor). It is **smooth, not fringes** → either
  genuinely non-Drude material response or a smooth frequency-dependent **mount-coupling bias**
  (pressed paper deforms the window/mount coupling: |C| CNT 0.85 vs Si 0.97–1.0). `C` can flag
  it but not correct it (the front/back split is unknowable — audited).

**VERDICT:** the analysis is at its **information limit**. No further analytical de-embed will
buy accuracy, because the remaining error is (i) a smooth mount-coupling systematic the data
cannot self-separate from material response, and (ii) closed-form conditioning at air
incidence behind *any* residual gap (F9/F19) — both physical/geometric, not algorithmic.
**The limiting factor is sample presentation, not analysis.**

**What survives robustly today (F28):** the mean gap (measured, reproducible); the
**anisotropy** (σ∥ > σ⊥ by ~2–7×, n∥/k∥ > n⊥/k⊥, robust in every fit variant); the p-pol
**window-frame trusted band ~0.9–2.2 THz** as the best single-spectrum view; and σ₁ in that
band is surprisingly robust to the coupling systematic (±10 % |H| scale → σ₁ = 52–62 S/cm at
1–2 THz, while n/k individually swing ±25 %).

---

## 9. Path forward and open questions

**Recommended endgame (F13, F15, F16, F28):** CNT deposited or pressed on a rigid **flat
HR-Si** substrate, measured in reflection *through* the Si. One jig attacks all three
problems: rigid substrate → mount coupling stays stable (Si controls prove rigid mounts hold
|C| ≈ 1); deposited/intimate contact → gap → 0; Si incidence (n = 3.4 ≫ SiO₂ 1.95) →
maximum conditioning headroom, low band reopened. Design the jig around Si specifically, with
a simultaneous alignment reference to the flat plane.

**Interim recipe on existing data (F28):** p-pol window-frame σ₁ over 0.9–2.2 THz with ±15 %
systematic error bars, phase-excess gap monitoring per mount, and a Si-slot control each
session.

**Open questions (lab notebook):**
- `OQ1` — Does a flat HR-Si jig deliver gap-free contact, or just move the gap? (F13)
- `OQ2` — Quantify how much a residual gap collapses the F7 conditioning advantage at Si
  incidence (Si-incidence + µm-gap model).
- `OQ3` — Process Denis's reflection dataset properly to settle its doping (F12).
- `OQ4` — Min-phase/MEM de-embed as the production gap estimator on the latest good-contact
  data (F11/F18).

---

## 10. Where to read more (map of the primary sources)

**Physics-decision log — `ANALYSIS_NOTES.md`:**
`§1` geometries/angles/sign · `§2`/`§13a` segmentation & GaP echoes · `§4` T0/phase ·
`§6`/`§6b` Si benchmark + window-vs-air conditioning · `§9`/`§9b` air gap + phase audit ·
`§11`/`§14` self-referencing + abs-time phase reference · `§15`/`§16`/`§17` windowing styles &
shared-axis isolation · `§18` transmission phase chain · KK + self_phase sections at the end.

**CNT findings log — `reports/CNT_measurement_lab_notebook.md`:** `F1`–`F29`, the
never-erased running record. **Start here for the current interpretation** (the Working
synthesis at the top is kept live).

**Tutorials — `reports/`:**
- `reflection_interface_theory_tutorial.md` — impedance, the pole, high-index rescue
  (pairs with `explorations/reflection_theory/drude_interface_model.py`).
- `lineshape_and_inversion_tutorial.md` — signal-vs-artifact lineshapes.
- `misalignment_lineshape_report.md` — how wavefront distortion → group delay → n spikes.
- `air_gap_deembed_research.md` — literature recanvas (min-phase/MEM/amplitude-KK/Si-anchor).
- `cnt_rough_gap_literature_canvas.md`, `gap_roughness_modelling_implementation_plan.md`.
- `uncertainty_propagation.md` — frequency-domain error model.

**Pipeline map — `PIPELINE_ARCHITECTURE.md`:** stage order, the `processing_dict` contract,
per-stage I/O, the `THzDataReflection` delegation nuance.

**Drivers:** `run_me_low-level.py` (live reflection pipeline), `run_me_dual_pol.py` (s/p),
`run_me_reflection_single.py` (bare gold-referenced), `run_me_transmission.py`.

**Explorations of note:** `explorations/air_gap_cnt_reflection/` (CNT-21 processing,
per-channel, slider explorer), `explorations/explore_air_gap_deembedding.py`,
`explorations/reflection_theory/`.
