# THz ellipsometry for highly conductive samples — methodology evaluation

*2026-09-30. Evaluation of full reflection ellipsometry (ρ = r_p/r_s) as the measurement route
for CNT buckypaper and other near-mirror samples, against the error budget we actually have.
Numbers from `explorations/thz_ellipsometry/ellipsometry_conditioning_analysis.py`
(638 lines, 3 known-answer checks pass). Companion to lab notebook F33.*

---

## 1. Verdict

**Do it, and do it at 70–80° incidence in free space with no window.** Ellipsometry is not a
marginal improvement on what we have — it removes, by construction, the single error channel
that has defeated every CNT campaign since F8, and it does so without giving up meaningful
sensitivity provided we go to high incidence angle.

Three results carry the decision:

1. **Every dominant term in our error budget is common-mode and cancels exactly in ρ.**
   A sample height error, a reference timing drift, the purge transient, lock-in gain, laser
   power — all multiply `r_p` and `r_s` by the *same* factor and divide out. In referenced
   reflectometry a **1.5 fs** timing error already costs Δ|N| = 0.81 on our CNT surrogate at
   1 THz/45°; 19 fs (the F19 "time-invisible" 4 µm gap) costs 23. In ellipsometry both cost
   **zero**, to machine precision.
2. **The replacement error — sample tilt — is ~500× weaker per unit of pain.** 0.2° of tilt
   costs Δ|N| = 0.11–0.16. To do as much damage as *1.5 fs* of reference timing you would need
   ≈ 1° of angle error. We have swapped a femtosecond-precision problem for a degree-precision
   problem.
3. **We are unusually well equipped for it.** The spintronic emitter is a polarization-state
   generator with no moving optics, and a `<110>` GaP electro-optic crystal is an analyser with
   *infinite* extinction ratio. Wire-grid polarizer quality is the accepted accuracy limiter of
   THz ellipsometry in the literature — and our setup doesn't need one.

The cost is real but bounded: a 70–80° reflection geometry (arms at 140–160°), a flat sample
mount, and a beam-divergence budget.

---

## 2. Why it works: the structural argument

Our whole problem is that the air|CNT reflection sits near the `r = −1` pole, so the inversion
amplifies *any* small multiplicative error. That amplification is a property of the material,
not of our technique — ellipsometry does not remove it. What ellipsometry removes is the
**errors being amplified**.

Write the measurement honestly, including everything we don't know:

```
S(α, φ, ω) = d(φ)ᵀ · T_out(ω) · J_sample(ω) · T_in(ω) · e(α) · E₀(ω) · g(ω)
```

`E₀` is the emitted spectrum, `g` any common gain/timing factor, `T_in`/`T_out` the unknown
Jones matrices of the emitter optics, off-axis parabolas and detector arm, `e(α)` the emitter
polarization state and `d(φ)` the EO detection projection.

- **Everything scalar (`E₀`, `g`) cancels in any ratio of two polarization channels.** This is
  the height/timing/drift immunity. It is exact, not approximate.
- **Everything matrix-valued (`T_in`, `T_out`, `d`, `e`) is *fixed* as long as nothing moves** —
  which is precisely the operating condition you identified as necessary. It therefore cancels
  against a single reference measurement, by the eigenvalue argument in §4.

That split is the whole case. The error channels that kill us are in the first group.

### The conditioning trade, quantified

Sensitivity to the index, for a conductor in the large-|N| limit, at incidence from medium n₁:

| observable | deviation from its limiting value | angular factor |
|---|---|---|
| `r_s` | `−1 + 2n₁cosθ/N` | `cosθ` — **dies at grazing** |
| `r_p` | `+1 − 2n₁/(N cosθ)` | `1/cosθ` — grows |
| `ρ = r_p/r_s` | `−1 + (2n₁/N)·sin²θ/cosθ` | `sin²θ/cosθ` — grows |

So `ρ` has `tan²θ` times the sensitivity of `r_s`, and `sin²θ` times that of `r_p` (→ 1 at
grazing). Ellipsometry gives up essentially nothing against the *best* single channel once
θ is large, and beats the channel we actually use by a factor that grows fast.

Measured on the CNT surrogate (Drude, σ_dc = 65 S/cm, τ = 30 fs, ε∞ = 4 — tuned to reproduce
our measured σ₁ ≈ 52–62 S/cm; gives n − iκ = 6.96 − 8.11i at 1 THz, |N| = 10.7):

Index error Δ|N| per **1% relative error in the observable**, 1 THz:

| θ | \|1+r_s\| | tanΨ | Δ | via `r_s` | via `r_p` | via `ρ` | ρ gain vs r_s |
|---|---|---|---|---|---|---|---|
| 45° | 0.127 | 0.918 | 174° | 0.808 | 0.404 | 0.808 | 1.0× |
| 65° | 0.077 | 0.793 | 164° | 1.353 | 0.243 | 0.296 | 4.6× |
| 70° | 0.063 | 0.737 | 159° | 1.672 | 0.197 | 0.224 | 7.5× |
| 75° | 0.048 | 0.660 | 150° | 2.209 | 0.151 | 0.162 | 13.6× |
| 80° | 0.032 | 0.553 | 132° | 3.293 | 0.107 | 0.110 | 29.8× |
| 85° | 0.016 | 0.463 | 85° | 6.562 | 0.081 | 0.082 | 80.3× |

**At 45° ellipsometry buys no sensitivity at all** — identical to `r_s`. The sensitivity win is
entirely a high-angle effect. At 45° the *only* win is error immunity (which is the dominant
term, so it is still worth doing — but do not expect the conditioning to improve).

---

## 3. Point 1 — "never realign": what cancels, what doesn't

### Cancels exactly
- Sample height / displacement along the surface normal (the air gap, F8/F9/F11/F19/F27).
- Reference-to-sample timing drift; delay-stage positioning error; the F32 purge transient.
- Laser power, lock-in sensitivity changes, detector gain, THz amplitude drift.
- **Scalar surface-roughness attenuation.** The Kirchhoff specular factor
  `exp(−(4πσ_rms cosθ/λ)²)` has no polarization dependence, so it divides out of ρ identically.
  Verified by attenuating `r_p` and `r_s` independently and re-inverting: Δ|N| ≈ 10⁻¹⁴ in ρ
  versus **0.82** in `r_s` for σ_rms = 5 µm at 70°. Note the `cosθ` — grazing incidence makes
  roughness *less* important, not more.

That last one deserves emphasis: it directly answers the "visible scatterer, THz reflector"
case you raised. To first order, roughness attenuation is invisible to ellipsometry.

### Does not cancel
- **In-plane tilt** (changes θ). 0.2° → Δ|N| = 0.11 (45°) to 0.16 (75°). Tolerable, mechanical,
  and measurable with an autocollimator or the specular return of a visible laser.
- **Out-of-plane tilt** (rotates the s/p frame). Generates *apparent* anisotropy: 0.1° gives
  |r_ps/r_ss| = 0.0030, 1° gives 0.030. This is the one that can be mistaken for physics — but
  it is a fittable nuisance azimuth in generalized ellipsometry, not an unknown systematic.
- **Beam divergence** — an angular *blur*, so it biases rather than averages out. Modelled as a
  Gaussian angular spread it costs Δ|N| = 0.024 (1° spread, 70°) rising to 0.23 at 85°. This is
  what stops us going to the pure-sensitivity optimum of ~85°.

### The sample-flatness solution you already have
The CNT paper is **opaque**: penetration depth 2.9 µm at 1 THz (5.6 µm at 0.3 THz), so a 30 µm
paper attenuates a double pass by 3.7×10⁻⁵. The back face is invisible.

**Therefore press the paper flat from behind, against a rigid optical flat, with nothing in the
beam.** The flattening problem and the optical path are completely decoupled. This is the clean
exit from the window/air-gap saga: we were putting a window in the beam to solve a mechanical
problem that has a mechanical solution once we no longer need a front-surface reference.

---

## 4. Eigenvalue calibration — you do not need to know the instrument

The strongest result in this evaluation, and the one that makes "never realign" into a formal
guarantee rather than an aspiration.

Take two input polarizations (two magnet settings) and two detection projections (two EO crystal
azimuths). The measurement is a 2×2 complex block `Y = D T_out J_s T_in E`. Measure a gold
mirror in the same geometry to get `G`. Then

```
Y G⁻¹  is similar to  J_s J_gold⁻¹
```

and **eigenvalues are invariant under similarity**. Their ratio is `ρ_sample / ρ_gold`, with
`ρ_gold` known from Fresnel to ~10⁻³. So ρ is recovered *without knowing* `T_in`, `T_out`, the
OAP cross-polarization, the emitter polarization impurity, the detection vector, or the
magnet-angle accuracy.

Numerically verified: injecting random 25% gain-imbalance / cross-talk / phase instrument
matrices, the eigenvalue route recovers ρ to **1×10⁻¹⁶**, while naively ratioing the diagonal
elements is wrong by 2–6%.

| θ | eigen error | naive error | eigenvalue separation |
|---|---|---|---|
| 45° | 1e-16 | 0.022 | 0.126 |
| 70° | 4e-16 | 0.053 | 0.410 |
| 80° | 1e-16 | 0.053 | 0.745 |

The eigenvalue *separation* is the conditioning of the calibration itself — small separation
means a noise-sensitive branch assignment. It improves strongly with angle: another reason for
70–80°.

**Three consequences worth internalising:**
- The magnet angle needs to be **repeatable, not accurate**. Whatever `e(α)` actually is, the
  gold measurement captures it. Same for an α-dependent emitter amplitude (incomplete
  saturation): with two fixed settings it is absorbed into `T_in` exactly.
- Swapping in the gold reference is **safe**, because what we extract from it (a polarization
  ratio) is itself immune to the placement error. This is the structural difference from a
  reflectometric reference, where the placement error *is* the measurement.
- The method is the established Eigenvalue Calibration Method (Compain, Poirier & Drevillon,
  *Appl. Opt.* **38**, 3490 (1999)); I have not found it applied to THz-TDS ellipsometry, which
  is a small novelty available to us.

---

## 5. Point 2 — spintronic emitter + GaP detection

### The emitter is a near-ideal polarization state generator
The ISHE emission is strictly perpendicular to **M**, so rotating the magnet rotates the THz
polarization *with no optic moving anywhere in the beam*. Nandi/Tiercelin et al.
([arXiv:2111.07118](https://arxiv.org/pdf/2111.07118), FeCo/TbCo₂/FeCo) demonstrate full 360°
rotation with a 5 kA/m (6.5 mT) hard-axis sweep, and — the part that matters spectroscopically —
show by Fourier analysis that the two field components stay **exactly in phase or antiphase
across the whole spectrum**: zero ellipticity, perfect linearity at every frequency, and no
amplitude loss on rotation. They explicitly propose this as a THz TD ellipsometer that
"would not require a rotating analyser anymore". It has since been done: *THz Time-Domain
Ellipsometry With Spintronic Emitters* ([IEEE](https://ieeexplore.ieee.org/abstract/document/10964355))
reports complete phase-resolved TDSE (full Jones and Mueller) from a spintronic source.

### The EO crystal is a perfect analyser — this is the underrated advantage
For a `<110>` zincblende crystal (Planken et al., *JOSA B* **18**, 313 (2001)):

```
S ∝ E_[001]·sin(2φ) + 2·E_⊥·cos(2φ)      φ = probe polarization angle from [001]
```

So the detection projection is the **real** vector `d(φ) = (sin 2φ, 2 cos 2φ)`. Three properties
follow, and each one deletes a calibration step that the THz ellipsometry literature treats as a
limiter:

1. **Real** — the detection channel ratio carries no phase, so a mis-set azimuth cannot corrupt
   arg(ρ), only |ρ|.
2. **Frequency-independent** — GaP is cubic, so both THz components see the same index,
   absorption and phase matching. `d_p/d_s = tan(2φ)/2` is one number, not a spectrum. Any
   frequency dependence we measure in it is a *diagnostic* of something else being wrong.
3. **Infinite extinction ratio** — the projection is defined by crystal symmetry, with no
   leakage term. Contrast the tutorial requirement of ER > 60 dB to hold a 1% amplitude error
   when |E_s| = 10|E_p|, against real free-standing wire grids that give |R| < 100 above
   0.6 THz. Section IV.C of the standard tutorial simply does not apply to us.

### Your "mid-angle" instinct is right — here is the number
Setting `tan 2φ = 2`, i.e. **φ = 31.72° between the probe polarization and the GaP [001] axis**,
gives `d = (0.894, 0.894)`: exactly equal sensitivity to both orthogonal THz components, at 45%
of the peak single-channel efficiency (|d| = 1.265 versus the 2.0 maximum). Across a full 90°
sweep of THz polarization the detected signal then varies only between 0.894 and 1.265 — i.e.
**71–100% of its maximum, never nulling.** That is precisely the behaviour you described.

| φ from [001] | d_[001] | d_⊥ | \|d\| | ratio |
|---|---|---|---|---|
| 0° | 0.000 | 2.000 | 2.000 | 0 |
| 22.5° | 0.707 | 1.414 | 1.581 | 0.50 |
| **31.72°** | **0.894** | **0.894** | **1.265** | **1.00** |
| 45° | 1.000 | 0.000 | 1.000 | ∞ |

### But one azimuth is not enough for anisotropy — see §6.

### The better detection scheme: spinning EO sampling
Rather than two discrete azimuths, rotate the GaP continuously. With the crystal at azimuth β
and THz at γ, the balanced signal is `ΔI ∝ E{½cos(β+γ) + (3/2)cos(3β−γ)}` — algebraically
identical to the Planken expression, but now β = ωt, so the ω and 3ω harmonics give **both field
components, amplitude and angle, in one acquisition, with no analyser**
([Opt. Express **28**, 13482 (2020)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-9-13482&id=431051);
a GaP/air-plasma version at [PMC9068238](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9068238/)).
Reported precision: **0.5° in polarization angle over 0.3–1.8 THz** after Jones calibration.
Their measured instrument matrix, `T = [[1, −0.003],[0.011, 0.80]]`, is worth looking at: a **20%
s/p amplitude imbalance** and ~1% cross-talk in an ordinary THz beam path. That is exactly the
`T_out` the eigenvalue method eliminates, and it shows the imbalance is not a small correction.

Caveat: a spinning crystal adds wobble/wedge-driven probe walk at the rotation frequency. Start
with two discrete azimuths (rotating the EO crystal about the beam axis at normal incidence
deviates nothing and leaves the THz path untouched); go spinning if the discrete version is
limited by drift between azimuths.

---

## 6. Point 3 — mixing, anisotropy, depolarization

### Polarization mixing: the rank result
With emitter rotation alone and a **single** detection azimuth, the measurement is
`S(α) = cosα·(d_p r_pp + d_s r_sp) + sinα·(d_p r_ps + d_s r_ss)` — a pure sinusoid in α. A full
360° rotation scan therefore yields exactly **two** complex numbers, no matter how finely you
sample it. Four Jones entries (three independent ratios) cannot be recovered from two.

Confirmed numerically: design-matrix rank **2/4** for one azimuth; **4/4** for two azimuths
(φ = 31.72° and 76.72°). So:

> **Emitter rotation alone is complete for isotropic samples and structurally rank-deficient for
> anisotropic ones.** Generalized ellipsometry needs a second detection projection. It does not
> need a rotating analyser — a second GaP azimuth, or spinning, is sufficient.

This is a real constraint on the CNT-21-style anisotropy work, and it is cheap to satisfy.

### Anisotropy: two cases, only one of which is hard
- **Randomly in-plane-oriented buckypaper** → uniaxial with the optic axis along the surface
  normal. The optic axis lies *in* the plane of incidence, so **no cross-polarization**: standard
  ellipsometry applies, and ρ separates the in-plane from the out-of-plane response. That is a
  bonus, not a problem — it is how Zhang et al. resolved in-plane vs out-of-plane resistivity of
  SWCNT films (*Carbon* **131**, 2018: doped 0.516±0.008 and 0.17±0.04 mΩ·cm; undoped 2.21±0.03
  and 17±1 mΩ·cm).
- **Aligned-fibre CNT** → biaxial. Put the fibre axis along s or p and it degenerates to the
  standard case (rotate the sample 90° for the other axis); at a general azimuth you need
  generalized ellipsometry, which then also *returns* the azimuth — turning the F24 orientation
  bookkeeping into a fitted quantity rather than an assumption.

⚠️ **Cross-check flagged:** those literature SWCNT in-plane conductivities are 452 S/cm (undoped)
to 1938 S/cm (doped) — **7–30× our measured 52–62 S/cm**. Buckypaper porosity plausibly explains
a large dilution factor, but this is exactly the kind of thing a four-point-probe DC measurement
on the actual paper would settle in an afternoon, and it would be a known-answer validation of
the inversion (extrapolate the fitted Drude σ₁ to DC and compare). Worth doing before the
ellipsometry build, not after.

### Depolarization: less of a problem than in the visible
Coherent field detection only sees the specularly-coherent mode; incoherently scattered light is
largely not in the detected mode at all. So we measure a *coherent Jones matrix with reduced
amplitude*, and the amplitude reduction cancels in ρ (§3). Mueller-matrix formalism, which
optical ellipsometry needs for rough surfaces, is not forced on us.

What survives is the second-order term: the polarization-dependent Rayleigh-Rice correction,
handled as a Bruggeman effective-medium roughness overlayer in the optical model. Standard
practice, and its known caveat applies — the fitted EMA thickness and fill fraction are effective
parameters, not the physical roughness profile.

---

## 7. Combined error budget and recommended geometry

Quadrature sum of the independent terms, CNT surrogate at 1 THz (|N| = 10.68):

**Realistic first build** (0.5% observable noise = our F28 floor, 0.1° angle, 1% channel gain,
1° divergence):

| θ | noise | tilt | gain | divergence | **total** | rel. |N| |
|---|---|---|---|---|---|---|
| 45° | 0.404 | 0.056 | 0.844 | 0.010 | **0.938** | 8.8% |
| 65° | 0.148 | 0.057 | 0.300 | 0.017 | **0.339** | 3.2% |
| 70° | 0.112 | 0.065 | 0.226 | 0.024 | **0.261** | 2.4% |
| 75° | 0.081 | 0.080 | 0.163 | 0.039 | **0.203** | 1.9% |
| 80° | 0.055 | 0.112 | 0.110 | 0.077 | **0.184** | 1.7% |
| 85° | 0.041 | 0.217 | 0.082 | 0.230 | **0.329** | 3.1% |

**Well-executed** (0.5% noise, 0.05° angle, 0.2% gain from a gold reference, 0.5° divergence):
0.125 at 70° (1.2%), **0.097 at 75° (0.9%)**, 0.084 at 80° (0.8%).

Compare with where we are now: F28 closed the CNT-21 budget at a **smooth ~10% mount-coupling
systematic** that no model could separate. Ellipsometry at 75° projects **~1%** — an order of
magnitude, and against a *different* and mechanically controllable error class.

Note which term dominates: not noise, not tilt — **the p/s channel gain**. Hence the gold
reference (which drops it ~5×) and hence high angle (which weakens it ~8× from 45° to 80°).

### Recommendation: **θ = 70–75°**, not the 80–85° the pure sensitivity plot prefers
The limiter is the footprint. For EFL 150 mm and a 30 mm collimated beam:

| θ | 0.5 THz | 1 THz | 2 THz |
|---|---|---|---|
| 70° | 11.2 mm | 5.6 mm | 2.8 mm |
| 75° | 14.7 mm | 7.4 mm | 3.7 mm |
| 80° | 22.0 mm | 11.0 mm | 5.5 mm |
| 85° | 43.8 mm | 21.9 mm | 10.9 mm |

At 80° and 0.5 THz the beam needs 22 mm of *flat* sample. Any waviness over that footprint is a
distribution of local incidence angles — the divergence term, but worse and un-modelled. 70–75°
keeps the footprint under ~15 mm while giving up only ~0.03 in Δ|N|. Maximise EFL as far as the
sample size allows (divergence falls as EFL², footprint only as EFL).

---

## 8. Honest limits — what we give up or must watch

- **No absolute reflectance.** ρ gives two real numbers (Ψ, Δ) → exactly enough for a bulk
  isotropic n, κ, and nothing spare. A supported thin film (n, κ, d) needs a second angle or
  added amplitude information. Fine for opaque buckypaper; not automatically fine for the
  deposited-on-HR-Si route of F13/F16.
- **At 45° there is no conditioning gain.** If the geometry is constrained to 45°, ellipsometry
  still wins on error immunity, but the pole-neighbourhood amplification is unchanged.
- **Mechanics.** 70–80° reflection means arms at 140–160° and a sample stage with sub-0.05°
  reproducibility, inside the purge enclosure. This is the main build cost, and F32 says
  anything that opens the enclosure costs ~3.7 h of re-equilibration — so design the sample
  exchange to work from outside, as you already did with the lateral translation stage.
- **OAP cross-polarization is not negligible.** Up to −8.5 dB locally inside the waist for a 90°
  OAP, ~1.5% net at the detector; symmetric OAP arrangements cancel much of the rotation. It is
  a fixed instrument term (eigenvalue-removable), but it eats into the dynamic range available
  for real anisotropy, so arrange the parabolas symmetrically.
- **Branch/root selection.** ρ → −1 for a good conductor and the closed-form inversion has the
  same root-picking hazards we hit in F25/F29/F30. Expect to reuse the global-branch-vote fix
  rather than per-bin passivity.
- **The CNT surrogate is a model.** Every number above is conditioned on a single-Drude
  |N| ≈ 10.7 at 1 THz. If the real paper is 7–30× more conductive (§6 cross-check), |N| rises by
  ~3–5× and all the sensitivity numbers scale roughly as |N| — the *relative* ranking of methods
  is unchanged, but the absolute budget gets worse and the argument for high angle gets stronger.

---

## 9. Literature map

**Tutorials / reviews**
- Chen & Pickwell-MacPherson, *An introduction to terahertz time-domain spectroscopic
  ellipsometry*, [APL Photonics **7**, 071101 (2022)](https://pubs.aip.org/aip/app/article/7/7/071101/2835194/An-introduction-to-terahertz-time-domain).
  The reference text. Explicitly identifies conductive films and "uncontrollable air gap"
  reflection as the failure modes ellipsometry solves. Angle-error propagation (Fig. 5),
  divergence table, extinction-ratio requirements, pulse-shift calibration (10 fs → 3.6°/7.2°/
  14.4° phase error at 1/2/4 THz — our F31 timing walk in their language).
- Neshat & Armitage, *Developments in THz range ellipsometry*, [arXiv:1305.3127](https://arxiv.org/pdf/1305.3127).

**Instrumentation & calibration**
- Neshat & Armitage, *THz-TDS ellipsometry: instrumentation and calibration*,
  [arXiv:1209.1294](https://ar5iv.arxiv.org/html/1209.1294). Variable angle 15–85° with a
  **fibre-coupled detector so the angle can change without realignment** — the same design
  instinct as yours. Gold-mirror detector-polarization-vector calibration; Johs regression
  calibration. Results: doped Si (ρ_DC = 0.0151 Ω·cm) at 73°; 1.95 µm SiO₂ recovered as 1.9 µm,
  0.3% of the 600 µm wavelength.
- *Accurate THz ellipsometry using calibration in time domain*, [Sci. Rep. **12** (2022)](https://www.nature.com/articles/s41598-022-10804-w).
- Compain, Poirier & Drevillon, *Appl. Opt.* **38**, 3490 (1999) — the eigenvalue calibration
  method (§4).

**Highly conductive regime**
- *THz-TDE with high precision for GaN with carrier densities up to 10²⁰ cm⁻³*,
  [Sci. Rep. **11** (2021)](https://www.nature.com/articles/s41598-021-97253-z) — the closest
  published analogue to our regime.
- *Characterization of ultrathin conductive films using a simplified approach for THz-TDSE*,
  [JIMTW (2024)](https://link.springer.com/article/10.1007/s10762-024-01011-x). Two ideas worth
  stealing: a **mirrored sample holder that turns a transmission-geometry TDS bench into an
  ellipsometer** (relevant if rebuilding the arms is too disruptive), and a single-interface
  sheet-conductivity model for films thinner than λ.
- Schubert group (Nebraska/Lund) — THz generalized and Mueller-matrix ellipsometry, 0.9–20 THz,
  polarizer-sample-rotating-analyser, 3×3 normalized Mueller block.

**Spintronic emitters for ellipsometry**
- Nandi/Tiercelin et al., [arXiv:2111.07118](https://arxiv.org/pdf/2111.07118) — 360° rotation,
  6.5 mT, perfect spectral linearity.
- *THz TD Ellipsometry With Spintronic Emitters: Pauli Coefficients…*,
  [IEEE (2025)](https://ieeexplore.ieee.org/abstract/document/10964355) — complete Jones/Mueller
  TDSE from a spintronic source.

**Detection**
- Planken et al., *JOSA B* **18**, 313 (2001) — `<110>` EO orientation dependence.
- *THz-TDP based on spinning E-O sampling: precision and calibration*,
  [Opt. Express **28**, 13482 (2020)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-9-13482&id=431051)
  and the GaP/air-plasma version, [PMC9068238](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9068238/).

**CNT-specific**
- Zhang et al., *Determination of conductivity anisotropy and the role of doping in SWCNT thin
  films with THz spectroscopic ellipsometry*, [*Carbon* (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0008622317312447).
- *THz transmission ellipsometry of vertically aligned MWCNTs*, [APL **101**, 111107 (2012)](https://pubs.aip.org/aip/apl/article-abstract/101/11/111107/127514/Terahertz-transmission-ellipsometry-of-vertically).

**Roughness / depolarization**
- Franta & Ohlídal, EMA vs Rayleigh-Rice for ellipsometric characterization of rough surfaces,
  [*Opt. Commun.* (2004)](https://sciencedirect.com/science/article/abs/pii/S0030401804012908).
- Cross-polarization in OAP trains — [IEEE (2022)](https://ieeexplore.ieee.org/document/9769500/).

---

## 10. Suggested sequence

1. **Four-point-probe the CNT paper.** Settles the §6 factor-of-10 question and gives the
   inversion a known answer. Cheap, do it first.
2. **Prototype in the existing 45° geometry.** Two magnet settings + two GaP azimuths + a gold
   mirror; validate the eigenvalue calibration on HR-Si (must return 3.418 flat). No mechanical
   rebuild, tests the entire measurement and analysis chain, and directly answers whether the
   F28 10% systematic is common-mode.
3. **Then decide on the 70–75° arm rebuild**, informed by (2).
4. Flat rigid backing for the paper, pressed from behind, nothing in the beam.
5. Analysis side: the pieces largely exist (`invert_nk_reflection` already does p-pol with the
   global branch vote; `reflection_gap` has the dual-pol machinery). What is new is a Jones/Stokes
   layer and the eigenvalue calibration — small, pure, and unit-testable.

**The honest framing of the whole thing:** we have spent since F8 trying to *measure* a µm-scale
gap phase to femtosecond accuracy. Ellipsometry does not measure it better — it arranges for the
answer not to depend on it.
