# THz ellipsometry for conductive films — a prospecting report

**S. Brooke, IMDEA Nanociencia — 30 September 2026**
*Group discussion document. Companion to the detailed internal evaluation in
`thz_ellipsometry_methodology_evaluation.md`; numbers reproduced by
`explorations/thz_ellipsometry/`.*

---

## 1. The proposition

We should build a **THz time-domain ellipsometer** out of the bench we already have, and use it
for conductive samples — starting with CNT buckypaper — in **bare reflection at 70–75°, with no
window and nothing pressed against the measured surface**.

The argument in one sentence: for the last several months our CNT measurements have been limited
not by noise but by our inability to reproduce the optical path between a reference and a sample,
and **ellipsometry measures a quantity that is mathematically blind to that entire class of
error**.

Our hardware is unusually well suited to it. The spintronic emitter rotates the THz polarization
with no optic moving anywhere in the beam, and the GaP electro-optic crystal is a polarization
analyser with *infinite* extinction ratio. The published THz-ellipsometry literature repeatedly
names wire-grid polarizer quality as the accuracy limiter of the technique — and we would not use
one.

---

## 2. Why our samples are hard (background for the group)

Three compounding problems, established over the CNT campaign:

1. **Near-mirror conditioning.** A conductor reflects with `r ≈ −1`, which is exactly where the
   Fresnel inversion `r → n` is singular. Small multiplicative errors are amplified enormously,
   and *worst at low frequency*, where the material is most metallic.
2. **The reference problem.** Reflection spectroscopy needs an absolute reference (a gold mirror,
   or a window front-face). Any difference in optical path between reference and sample — a
   micron of height, a femtosecond of drift — enters as a multiplicative phase error and is then
   amplified by (1). We measured this: a **4 µm air gap is invisible in the time trace** (19 fs,
   below our centring resolution) yet drags the recovered index by Δn ≈ 0.8.
3. **Presentation.** Buckypaper is rough, porous and floppy. Pressing it against a window to
   flatten it creates the very gap in (2). Our error budget closed at a smooth **~10% systematic**
   attributable to mount coupling, which no amount of modelling could separate from real physics.

Transmission is not an escape: the samples are opaque (field penetration ≈ 3 µm at 1 THz).

---

## 3. What ellipsometry measures — the theory you need

### 3.1 The ellipsometric ratio

At oblique incidence, light splits into two independent polarizations: **s** (electric field
perpendicular to the plane of incidence) and **p** (in the plane). A surface reflects them
differently, with complex Fresnel coefficients

$$r_s = \frac{N_1\cos\theta_1 - N_2\cos\theta_2}{N_1\cos\theta_1 + N_2\cos\theta_2}, \qquad
r_p = \frac{N_2\cos\theta_1 - N_1\cos\theta_2}{N_2\cos\theta_1 + N_1\cos\theta_2}$$

where $N = n - i\kappa$ is the complex refractive index and $N_1\sin\theta_1 = N_2\sin\theta_2$.

Ellipsometry does not measure either coefficient. It measures their **ratio**:

$$\rho \;=\; \frac{r_p}{r_s} \;=\; \tan\Psi\,e^{i\Delta}$$

$\tan\Psi$ is the amplitude ratio, $\Delta$ the phase difference between the two polarizations.
For a bulk, isotropic, opaque sample this is exactly enough information — two real numbers per
frequency for two unknowns $(n, \kappa)$ — and the inversion is closed-form:

$$N_2 = N_1\sin\theta_1\sqrt{1 + \tan^2\theta_1\left(\frac{1-\rho}{1+\rho}\right)^2}$$

**No reference sample appears anywhere.** The sample references itself, because both polarizations
travel the same path through the same optics and hit the same spot at the same instant.

### 3.2 Why that kills our error class

Write any real measurement as the true reflection times everything we do not control:

$$r_{\text{measured}} = r_{\text{true}} \times \underbrace{g(\omega)}_{\text{gain, drift, timing, height}}$$

A sample sitting 2 µm too far back, a delay stage 10 fs out, the purge still equilibrating, the
lock-in sensitivity changed between runs — all of these are a *scalar* $g$. And a scalar multiplies
$r_p$ and $r_s$ **identically**, so it divides out of $\rho$ exactly:

$$\rho = \frac{r_p\,g}{r_s\,g} = \frac{r_p}{r_s}$$

This is not an approximation or a first-order cancellation. It is why the technique exists.

**One important caveat, developed in §4.3:** this argument covers errors common to *both
polarizations at the same instant*. If the p and s data are acquired at different times and the
instrument drifts in between, that differential error does **not** cancel. It is the one
first-order channel ellipsometry leaves open, and it dictates the acquisition protocol.

What it costs: $\rho$ carries no absolute reflectance. For an opaque bulk sample we do not need
one. For a thin film on a substrate (three unknowns: $n$, $\kappa$, $d$) we would need a second
incidence angle.

### 3.3 The sensitivity trade — and why the angle matters more than anything

For a good conductor ($|N| \gg 1$), expanding the Fresnel coefficients:

| quantity | behaviour | angular weight |
|---|---|---|
| $r_s$ | $-1 + 2N_1\cos\theta/N$ | $\cos\theta$ — **dies at grazing** |
| $r_p$ | $+1 - 2N_1/(N\cos\theta)$ | $1/\cos\theta$ |
| $\rho$ | $-1 + (2N_1/N)\sin^2\theta/\cos\theta$ | $\sin^2\theta/\cos\theta$ |

So $\rho$ carries $\tan^2\theta$ times the information about $N$ that $r_s$ does. Computed for a
Drude model matched to our measured CNT conductivity ($|N| = 10.7$ at 1 THz), as the index error
produced by a **1% error in the measured quantity**:

| angle | via $r_s$ (referenced) | via $\rho$ (ellipsometry) | advantage |
|---|---|---|---|
| 45° | 0.81 | 0.81 | **1.0× — none** |
| 65° | 1.35 | 0.30 | 4.6× |
| 70° | 1.67 | 0.22 | 7.5× |
| 75° | 2.21 | 0.16 | 13.6× |
| 80° | 3.29 | 0.11 | 29.8× |

**This is the single most important design number in the report.** At 45° — our current geometry —
ellipsometry buys error immunity and *no sensitivity whatsoever*. The technique only pays its way
at high incidence angle. Anyone tempted to test the idea at 45° and judge it on the conditioning
should know that in advance.

### 3.4 Anisotropic samples: the Jones matrix

An isotropic surface does not mix s and p. An anisotropic one does, and the description becomes a
**Jones matrix**:

$$\begin{pmatrix}E_p^{\text{out}}\\ E_s^{\text{out}}\end{pmatrix} =
\begin{pmatrix} r_{pp} & r_{ps} \\ r_{sp} & r_{ss}\end{pmatrix}
\begin{pmatrix}E_p^{\text{in}}\\ E_s^{\text{in}}\end{pmatrix}$$

The off-diagonal terms **are** the anisotropy signal. Measuring them is "generalized ellipsometry";
measuring only $\rho = r_{pp}/r_{ss}$ is "standard ellipsometry". Reciprocity ties $r_{sp}$ to
$r_{ps}$, so a single angle of incidence yields **two independent complex ratios**
($r_{pp}/r_{ss}$ and $r_{ps}/r_{ss}$) — four real numbers per frequency. That number governs
everything in §6.

---

## 4. What the experiment actually is

### 4.1 Hardware — what we already have

| element | role | what moves |
|---|---|---|
| Spintronic emitter + magnet | polarization state generator | the **magnet only**, never an optic |
| Sample, bare reflection, 70–75° | the measurement | nothing, once mounted |
| GaP `<110>` EO crystal | polarization analyser | rotation about the beam axis (no beam deviation) |
| Gold mirror in the sample slot | calibration | swapped once per session |

**Emitter.** The inverse-spin-Hall emission is strictly perpendicular to the magnetization, so
rotating the magnet rotates the THz polarization. Published work on anisotropy-engineered emitters
demonstrates a full 360° rotation with a 6.5 mT field sweep, with the two field components staying
exactly in phase or antiphase across the whole spectrum — perfect linear polarization at every
frequency, no amplitude loss on rotation.

**Detector.** For a `<110>` zincblende crystal the sampled signal is

$$S \;\propto\; E_{[001]}\sin 2\varphi \;+\; 2\,E_{\perp}\cos 2\varphi$$

with $\varphi$ the probe polarization angle from the crystal `[001]` axis. The detection is
therefore a projection onto a **real** vector $\mathbf{d}(\varphi) = (\sin 2\varphi,\,2\cos 2\varphi)$.
Three consequences, each of which deletes a calibration step that the literature treats as a limiter:

- **Real** — a mis-set crystal azimuth perturbs $|\rho|$ but cannot corrupt $\Delta$.
- **Frequency-independent** — GaP is cubic, so both THz components see the same index, absorption
  and phase matching. The channel ratio $d_p/d_s = \tan(2\varphi)/2$ is *one number, not a spectrum*.
  Any frequency dependence we measure in it is a diagnostic that something else is wrong.
- **Infinite extinction ratio** — the projection is defined by crystal symmetry. Real free-standing
  wire grids manage an intensity ratio below 100 above 0.6 THz, against a requirement of >10⁴ for
  1% accuracy in the unfavourable channel.

Setting $\tan 2\varphi = 2$, i.e. **$\varphi = 31.72°$ between the probe polarization and the GaP
`[001]` axis**, gives $\mathbf{d} = (0.894, 0.894)$: equal sensitivity to both components at 45% of
peak efficiency. Across a full 90° sweep of THz polarization the signal then varies only between
71% and 100% of maximum and never nulls.

### 4.2 What we vary and what we record

A complete measurement is a small grid:

- **Emitter polarization** $\alpha$: **6–12 steps over 180°**, not the two that p and s strictly
  require. The reason is in §4.3 and it is worth real money.
- **Detector azimuth** $\varphi$: **two settings** (e.g. 31.72° and 76.72°). This is the
  non-obvious requirement — see §6.
- **Delay scan** at each combination: the usual time-domain waveform.
- **Interleaved acquisition order** — cycle through all the settings repeatedly rather than
  finishing one before starting the next. Also §4.3.
- **Gold mirror**, same grid, once per session.

Everything else — sample, mirrors, purge, alignment — is untouched throughout.

### 4.3 Over-determine the polarization series — the one error ellipsometry does *not* kill

Agulto *et al.* (*Sci. Rep.* **11**, 18129, 2021) measured GaN at carrier densities to
10²⁰ cm⁻³ — our regime — and reported **ten times better precision** in the ellipsometric
parameters by recording 24 analyser angles and fitting, rather than just measuring p and s
directly, **at equal total measurement time**. That last clause is the important one: at fixed
time, extra angles buy essentially nothing by averaging. The gain is entirely a *systematic*
correction, and it is worth understanding exactly which systematic.

**The physics.** At any fixed point in the waveform, the signal as a function of the rotation
angle must follow a known low-parameter harmonic — for their rotating analyser,
$A\cos 2\theta_B + B\sin 2\theta_B + C$; for our rotating emitter, a first harmonic
$P\cos\alpha + Q\sin\alpha$. Anything that departs from that form is not the sample. They
attribute the departure to per-waveform timing jitter and time-shift each waveform to remove it.

**Why we should care, given §3.2 said everything cancels.** A timing error *common* to both
polarizations cancels exactly in $\rho$. A timing error *differential between the acquisitions
at different polarization settings* does not — it enters $\rho$ directly as
$e^{i\omega\,\delta t}$. **This is the one first-order error channel that ellipsometry does not
remove for free**, and it is a real refinement of the argument in §3.2. At 1 THz a 1.5 fs
differential drift is 0.94% in $\rho$, which on its own is comparable to the entire rest of the
error budget in §4.5.

We already know we have this: within-run timing walk was measured at ~17 fs/hour.

**But our error model is not theirs, and that changes the right correction.** Their model is
independent per-waveform jitter. Ours is a slow, ordered drift. Simulating our geometry at 1 THz
with 0.5% noise and equal total measurement time for every strategy (median index error):

| drift across the run | 2 settings, sequential | 2 settings, interleaved | 12 settings, no correction | 12 settings + per-setting delay (their method) | 12 settings + **one-parameter drift ramp** |
|---|---|---|---|---|---|
| 0 fs | 0.186 | 0.191 | 0.173 | 0.282 | **0.178** |
| 5 fs | 0.378 | 0.195 | 0.189 | 0.280 | **0.177** |
| 20 fs | 1.308 | 0.193 | 0.390 | 0.277 | **0.177** |
| 60 fs | 3.259 | 0.243 | 1.141 | 0.272 | **0.173** |
| 200 fs | 6.689 | 0.599 | 4.226 | 0.279 | **0.170** |

Five things fall out of this, and they are the acquisition protocol:

1. **Never run the polarization settings sequentially.** Measuring all of p and then all of s is
   the worst option available and degrades by a factor of 7 at only 20 fs of drift. This is the
   naive protocol.
2. **Interleaving is nearly free and does most of the work** — it holds ~0.19 out to 60 fs of
   drift. Same conclusion we reached for the bare/doped comparison previously, for the same
   reason.
3. **Over-determination alone does not save you** (12 settings with no correction still degrades
   to 1.14 at 60 fs). You have to actually use the redundancy.
4. **Their per-waveform delay fit is genuinely drift-immune but pays a constant penalty** —
   0.28 regardless of drift, against 0.17 for the best. It spends $N-1$ degrees of freedom where
   our error model needs one.
5. **A one-parameter linear drift ramp is the best of everything, in every regime** — 0.17, flat
   from 0 to 200 fs of drift, with no penalty at zero drift. **This is the improvement on their
   method, and it is available to us precisely because we know our timing error is a slow drift
   rather than white jitter.**

**Honest calibration of the "10×".** It is real, but it is a statement about their instrument's
jitter level, not a universal factor. Reproducing 10× in our simulation requires ~20 fs of
differential timing error. If drift is under control the multi-angle gain shrinks toward 1×. The
ramp-fitted version is nevertheless strictly better than every alternative at every drift level,
so adopt it regardless — it costs one fitted parameter and no measurement time.

**Free diagnostic.** The residual of the harmonic fit is a run-time quality flag that needs no
reference and no model of the sample: the signal *must* be a pure first harmonic in the emitter
angle, so whatever is left is instrument error, and we can size it without knowing its cause.
Fit a per-setting gain alongside the delay and see which the residual prefers, rather than
assuming — a residual that neither explains is an alarm worth halting on.

**What we should not copy.** Their rotating element is a wire-grid analyser physically turning in
the THz beam, with its own imperfections rotating with it, and their PCAs sit at −45° behind
polarizers to work around antenna polarization impurity. We get the same redundancy by turning a
magnet outside the beam, and our analyser is the EO crystal. Same information, no added optic.

**One structural note.** Their scheme (fixed input, rotating analyser) and ours (rotating input,
fixed projection) are **duals and carry identical information** — both yield two complex numbers,
i.e. rank 2 of the 4-element Jones matrix. Their instrument could not measure our anisotropy
either. The second detection azimuth in §6 is required either way.

### 4.4 The calibration, and why it is remarkably forgiving

Real THz beam paths are not polarization-neutral: a published Jones-matrix characterization of an
ordinary parabolic-mirror path found a **20% s/p amplitude imbalance** and ~1% cross-talk. Off-axis
parabolas generate cross-polarization up to −8.5 dB locally within the beam waist.

We do not need to know any of it. With two input polarizations and two detection azimuths the
measurement is a 2×2 complex block $Y$. Measuring gold in the same geometry gives $G$. Then

$$Y\,G^{-1} \quad\text{is \emph{similar} to}\quad J_{\text{sample}}\,J_{\text{gold}}^{-1}$$

and **eigenvalues are invariant under similarity**. Their ratio is $\rho_{\text{sample}}/\rho_{\text{gold}}$,
with $\rho_{\text{gold}}$ known from Fresnel to ~0.1%. In simulation, injecting random instrument
matrices with 25% gain imbalance and cross-talk, this recovers $\rho$ to **1 part in 10¹⁶**, where
naively ratioing the measured channels is wrong by 2–6%.

Three practical consequences worth stating plainly:

- **The magnet angle must be repeatable, not accurate.** Whatever the emitter actually emits, the
  gold measurement captures it.
- **Swapping in the gold reference is safe**, because what we extract from it is a *polarization
  ratio*, which is itself immune to placement error. This is the structural difference from a
  reflectometric reference, where the placement error *is* the measurement.
- The method is the Eigenvalue Calibration Method, standard in visible Mueller polarimetry since
  1999. I have not found it applied to THz time-domain ellipsometry.

### 4.5 Geometry: bare reflection, no window

Start with the simplest possible arrangement: **paper pressed as flat as possible, measured in
free space, nothing in the beam.**

The sample is opaque — field penetration 2.9 µm at 1 THz, so a 30 µm paper attenuates a double
pass by 3.7 × 10⁻⁵ and the back face is invisible. **Therefore flatten it from behind, against a
rigid optical flat.** The flattening problem and the optical path are completely decoupled. This is
the clean exit from the window/air-gap saga: we were putting a window *in the beam* to solve a
*mechanical* problem, and once we stop needing a front-surface reference, the mechanical problem
has a mechanical solution.

**On surface non-planarity** — the distinction matters and is worth being precise about. A wavy
surface does two different things:

- It **steers the beam**. This changes how much power couples into the detected mode. That is a
  scalar, so it cancels in $\rho$ and the EO crystal genuinely does not care. ✔
- It **tilts the local surface facet**, which is not scalar. In-plane tilt changes the incidence
  angle (a bias, mathematically the same as beam divergence). Out-of-plane tilt rotates the local
  s/p frame and **converts polarization** — generating fake off-diagonal Jones terms that mimic
  anisotropy. ✘

Quantitatively: a **systematic** out-of-plane tilt of 0.1° produces $|r_{ps}/r_{ss}| = 0.003$;
1° produces 0.030. The genuine anisotropy signal from our CNT model is 0.074 (§6). So **keep
systematic out-of-plane tilt below ~0.2°** and the artefact stays under 10% of the signal.
*Random* waviness with a symmetric tilt distribution averages toward zero in the off-diagonal term
and shows up as depolarization instead — annoying but not biasing. Useful diagnostic: the
tilt-induced cross-polarization is spectrally flat, whereas real Drude anisotropy is not.

For the diagonal channel, tolerances are comfortable: 0.2° of tilt costs Δ|N| ≈ 0.11–0.16, against
0.81 for a **1.5 fs** timing error in the referenced measurement. We are trading a femtosecond
problem for a degree problem.

### 4.6 Projected error budget

At 75°, with our measured 0.5% instrument floor, 0.05° angle reproducibility, a gold-calibrated
channel ratio and 0.5° of beam divergence:

| term | Δ\|N\| |
|---|---|
| measurement noise | 0.081 |
| angle / tilt | 0.040 |
| channel gain | 0.033 |
| beam divergence | 0.010 |
| **total (quadrature)** | **0.097 = 0.9% of \|N\|** |

Against the **~10%** smooth systematic that currently limits us. The dominant term is the channel
gain, which is why the gold reference matters and why high angle helps (it weakens ~8× from 45°
to 80°).

**Recommended angle: 70–75°, not the 80–85° that the sensitivity curve alone prefers.** The
limiter is footprint. With a 150 mm focal length and a 30 mm collimated beam, the illuminated patch
along the plane of incidence is 11 mm at 70°, 15 mm at 75°, but 22 mm at 80° and 44 mm at 85° (all
at 0.5 THz). Every millimetre of that has to be flat.

---

## 5. Frequency coverage — the case for dropping the window

Removing the window is not only about the gap. Fused silica absorption climbs steeply through the
THz band, and a double-passed few-millimetre window is what has been truncating our usable
bandwidth above ~2 THz. Bare reflection opens the high end, and the high end is where the
information is.

Two distinct claims, which should not be conflated:

1. **Absolute conditioning improves with frequency.** The index error per 1% observable error, at
   the optimum angle, falls from 0.143 at 0.3 THz to 0.048 at 3 THz — a factor of 3. The
   pole distance $|1+r_s|$ widens correspondingly.
2. **Relative conditioning is roughly flat.** Expressed as a fraction of $|N|$, the error is
   0.73% at 0.3 THz and 0.85% at 3 THz. The sample is less metallic at high frequency, so $|N|$
   falls in step. Going high does not, by itself, make the *fractional* index more accurate.

So where is the gain? **In the model parameters.** A Fisher analysis of a Drude fit to $\rho$, at
0.5% error, gives the 1σ uncertainty on the scattering time per octave:

| octave | δσ_dc/σ_dc | δτ/τ |
|---|---|---|
| 0.25–0.5 THz | 0.47% | **6.4%** |
| 0.5–1.0 THz | 0.36% | 2.4% |
| 1.0–2.0 THz | 0.28% | 1.0% |
| 2.0–4.0 THz | 0.22% | **0.44%** |

A **15× improvement** in the scattering time from the top octave over the bottom one — because
$\tau$ enters the Drude response through $\omega\tau$, and at 0.3 THz with $\tau \sim 30$ fs we are
deep in the flat part of the curve where $\tau$ is nearly invisible.

And the decisive number for the proposed strategy: fitting the **1.0–3.0 THz band alone** gives
δτ/τ = 0.52%, against 0.60% for the full 0.3–3.0 THz band. **Discarding the badly conditioned low
frequencies entirely costs essentially nothing.** Samuel's argument holds, with the caveat that the
mechanism is "the high band carries the $\tau$ information", not "the high band is fractionally
better conditioned".

(If the real scattering time is longer — 100 fs, knee at 1.6 THz — the high band becomes even more
dominant, δτ/τ = 0.32% from 1–3 THz.)

---

## 6. Anisotropic samples — do we have to rotate the sample?

This is the question the group should focus on, and it has a clean answer with one caveat.

### 6.1 The rank result

With the emitter rotating and a **single** detection azimuth, the detected signal is

$$S(\alpha) = \cos\alpha\,(d_p r_{pp} + d_s r_{sp}) + \sin\alpha\,(d_p r_{ps} + d_s r_{ss})$$

— a pure sinusoid in $\alpha$. A full 360° polarization series therefore yields **exactly two
complex numbers, no matter how finely it is sampled.** The design matrix has rank 2 of 4. Extra
angular steps buy signal-to-noise and let us fit the emitter's offset; they buy **no new
information about the sample.**

> **Answer to the question as posed: a polarization series alone does *not* substitute for
> rotating the sample. But rotating the sample is not the fix either — the fix is to rotate the
> *detection*.** A second GaP azimuth restores the design matrix to full rank 4 and delivers the
> complete normalized Jones matrix from a single mount, with the sample never touched.

### 6.2 Mount the fibre axis at ~45°

Cross-polarization vanishes when the sample's optic axis lies along s or p. Computed with a full
anisotropic (Berreman-class) reflection solver for our CNT model at 1 THz, 70°:

| sample azimuth | \|r_ps/r_ss\| |
|---|---|
| 0° | 0.0000 |
| 22.5° | 0.0563 |
| **45°** | **0.0739** |
| 60° | 0.0608 |
| 90° | 0.0000 |

At 0° or 90° the anisotropy is **invisible from a single mount** — you would have to rotate the
sample. At 45° the signal is 7.4%, comfortably above the ~1.5% instrumental cross-talk and the
0.3% from 0.1° of tilt. **Mount the fibre axis at 45° to the plane of incidence.**

### 6.3 What is and is not recoverable

A single angle of incidence gives four real observables per frequency against seven unknowns
(three complex principal permittivities plus the azimuth). **Point-by-point model-free inversion
of the full anisotropic tensor is impossible** — that is a counting result, not a limitation of
our instrument. Close it one of three ways: a parametric model across frequency (a Drude with a
handful of parameters is enormously over-determined), an assumption of transverse isotropy, or a
second incidence angle.

Monte Carlo, shared Drude over 0.6–3.0 THz, 0.5% noise, single mount at 45°:

| parameter | true | recovered (1σ, 24 trials) | relative |
|---|---|---|---|
| σ along fibre | 65.0 S/cm | 65.3 ± 1.7 | 2.7% |
| σ across fibre | 16.2 S/cm | 16.3 ± 0.2 | 1.2% |
| σ out of plane | 6.5 S/cm | 6.6 ± 0.2 | 2.8% |
| τ | 30 fs | 29.8 ± 0.4 | 1.2% |
| azimuth | 45° | 45.1 ± 0.6° | — |

All unbiased, and all five parameters are recovered — including the out-of-plane component, which
I expected beforehand to fail. It does not, because although its Jacobian is only 0.28× the
in-plane one at this angle, 3.6× worse conditioning against a 1.2% baseline is still a usable 2.8%.

Two honest caveats on that row. First, this simulation fits *exactly* the model that generated the
data, and the weakest-constrained parameter is always the one that absorbs model error — so I would
not quote an out-of-plane conductivity from a single incidence angle without a second, more grazing
angle to corroborate it. Second, the *less* conductive in-plane direction is recovered better than
the more conductive one (1.2% vs 2.7%), which is the same near-mirror conditioning story as
everywhere else in this report.

The azimuth comes out as a *fitted quantity* to ±0.6°, which converts the sample-orientation
bookkeeping that caused us trouble in the previous campaign into a measured number.

### 6.4 The honest caveat on rotating the sample anyway

In an idealized comparison where the two mounts are otherwise identical, physically rotating the
sample to 0° and 90° and running two *standard* ellipsometry measurements is dramatically more
precise than the single-mount generalized route — roughly a hundredfold on the anisotropy ratio.
Each mount then isolates a principal axis cleanly, with no cross-polarization to measure and no
azimuth to fit. That is a real statistical advantage and we should not pretend otherwise.

But that comparison assumes the re-press is free. For us it is not.

### 6.5 The comparison with realistic mount-to-mount variation

Repeating it with an **unmodelled per-mount nuisance** — 0.5° of tilt and 2% of coupling change,
drawn independently for each mounting operation, which is what re-pressing a sheet of buckypaper
actually does:

| scheme | mounts | σ along | σ across | anisotropy ratio |
|---|---|---|---|---|
| generalized, one mount | 1 | 67.5 ± 9.0 | 16.4 ± 0.9 | 4.11 ± 0.33 |
| standard, two mounts | 2 | 66.3 ± 3.3 | 16.6 ± 0.7 | 4.10 ± 0.27 |
| *true* | | *65.0* | *16.2* | *4.00* |

**The hundredfold advantage collapses to less than a factor of three, and on the anisotropy ratio
— the quantity we actually care about — to about 20%.** Both routes pick up a small positive bias
from the nuisance. (8 trials only, so the spreads are themselves uncertain at the tens-of-percent
level; treat this as an order-of-magnitude statement, not a precise one.)

Worth noting explicitly: this penalty is *smaller* than it would have been in the windowed
geometry, because with no window there is no contact gap to change between mounts. Removing the
window improves the two-mount route as well as the one-mount route.

**Recommendation: single-mount generalized measurement as the primary**, on the grounds that it
delivers comparable precision with one mounting operation instead of two, never disturbs the
sample, and returns the azimuth as a fitted quantity that validates itself. **Use the 90° rotation
as a cross-check, not as the method.** If the two agree, the mount systematic is under control. If
they disagree, we have just measured it — which is a useful result in its own right, and precisely
the measurement we could not make in the windowed geometry.

---

## 7. Literature landscape, and the gaps we can fill

*Caveat for the group: this is a targeted survey, not a systematic review. Novelty claims below
should be re-checked before anything is written up.*

### 7.1 What is already established — we would not be first at any of this

| area | status |
|---|---|
| THz-TDS ellipsometry as a technique | Established since Nagashima & Hangyo (2001); a full tutorial exists (Chen & Pickwell-MacPherson, *APL Photonics* **7**, 071101, 2022) |
| Variable-angle THz-TDSE instrumentation and calibration | Neshat & Armitage, *Opt. Express* (2012) — 15–85°, fibre-coupled detector specifically so the angle can change without realignment, gold-mirror calibration, regression calibration |
| Generalized / Mueller-matrix THz ellipsometry | Schubert group (Nebraska/Lund), mostly frequency-domain, 0.9–20 THz, rotating analyser |
| Highly conductive samples by THz ellipsometry | Doped Si (ρ = 0.015 Ω·cm); GaN to 10²⁰ cm⁻³ (Agulto *et al.*, *Sci. Rep.* **11**, 18129, 2021) at 70° incidence — and they independently report that tanΨ → 1 and Δ → π as carrier density rises, i.e. our ρ → −1 conditioning problem |
| Over-determined polarization series to remove timing systematics | Agulto *et al.* (2021), 24 analyser angles, per-waveform jitter correction, 10× precision at equal measurement time |
| CNT films by ellipsometry | SWCNT thin films on substrates, THz–UV, in-plane vs out-of-plane resistivity (*Carbon*, 2018); vertically aligned MWCNTs in transmission (*APL* **101**, 111107, 2012) |
| Spintronic emitter polarization control | 360° rotation with a 6.5 mT sweep (arXiv:2111.07118) |
| Spintronic emitters *in* ellipsometry | Done — complete Jones/Mueller TDSE (IEEE, 2025). The field is one or two papers deep |
| Spinning electro-optic polarimetry | *Opt. Express* **28**, 13482 (2020); GaP version exists |
| Eigenvalue calibration method | Standard in visible Mueller polarimetry since Compain & Drevillon (1999) |
| Roughness in ellipsometry | Mature, but a visible-range literature (EMA vs Rayleigh–Rice) |

### 7.2 Where the gaps are

**(a) An ellipsometer with no polarizing optics and no moving parts in the THz beam.**
Published THz-TDS ellipsometers use photoconductive antennas plus two or three wire-grid
polarizers, and the reviews consistently identify polarizer extinction ratio as *the* accuracy
limiter. A spintronic emitter supplies the polarization state generator by symmetry, and a `<110>`
EO crystal supplies the analyser by symmetry. Combining both removes every polarizer from the
instrument. The spintronic-ellipsometry work that exists appears still to analyse on the detection
side conventionally; the combination looks open.

**(b) Eigenvalue calibration at THz.** Twenty-five years old in the visible, and exactly the right
tool for a band where the beam path demonstrably imposes a 20% s/p imbalance and percent-level
cross-talk. I found no THz-TDS application. This is a small, clean, checkable methods contribution.

**(b′) A drift-matched nuisance model for the over-determined polarization series.** The published
correction (§4.3) fits an independent delay per waveform, which is right for white jitter and
wasteful for a slow drift. Replacing $N-1$ free delays with a one-parameter ramp is strictly
better in our simulations at every drift level, including zero. Small, but it is a real
improvement on the state of the art and costs nothing to implement.

**(c) Reflection generalized ellipsometry of free-standing, rough, near-mirror conductors.**
The conductive-sample ellipsometry literature is polished bulk semiconductors — flat, rigid, clean.
Buckypaper is the opposite: rough, porous, anisotropic, floppy, opaque in transmission. The CNT
ellipsometry that exists is films *on substrates* or vertically aligned forests *in transmission*.
Free-standing paper in reflection is where our samples actually live, and it is close to empty.

**(d) A quantified error-budget case for ellipsometry on near-mirror samples.** The reviews assert
the self-referencing advantage qualitatively. We can state it quantitatively — the common-mode
cancellation, the $\tan^2\theta$ crossover (no gain at 45°, 13.6× at 75°), and the trade of a
femtosecond tolerance for a degree tolerance — against a *measured* error budget. That is directly
reusable by anyone else measuring conductors and is a natural methods section.

**(e) Single-mount generalized ellipsometry for soft anisotropic samples.** The identifiability
analysis in §6 — what one mount with two detection projections can and cannot deliver, and why
mounting at 45° is mandatory — is a practical result for any sample that cannot be re-mounted
reproducibly.

**Realistic framing.** (a), (b) and (e) are instrument and method contributions of modest but real
novelty. (c) is the science we actually want. (d) is the argument that ties them together. None of
it requires us to be first at ellipsometry — only first at this *combination*, on *these* samples.

---

## 8. Risks and stopping conditions

| risk | mitigation | would stop us |
|---|---|---|
| Paper not flat enough over an 11–15 mm footprint | rigid backing plate, pressed from behind; reduce angle to shrink the footprint | if flatness cannot get under ~0.2° systematic tilt, cross-polarization is contaminated and anisotropy is unreliable (the diagonal channel would still work) |
| 70–75° geometry disruptive to build | prototype first at the existing 45°, which needs no rebuild | if a two-arm 140–160° layout cannot fit in the purge enclosure |
| Purge equilibration | measured at ~3.7 h to 99%; design sample exchange to work from outside the box | — |
| OAP cross-polarization eats the anisotropy dynamic range | symmetric parabola arrangement; eigenvalue calibration removes the mean term | if residual cross-talk exceeds ~3% it competes with the 7.4% signal |
| Differential timing drift between polarization settings | interleave; fit a one-parameter drift ramp; log the harmonic residual (§4.3) | if residual drift after correction exceeds ~1.5 fs it is comparable to the whole rest of the budget |
| Root/branch selection in the inversion | $\rho \to -1$ for conductors; reuse the global-branch-vote fix from the previous campaign | — |
| Our CNT conductivity may be wrong | four-point probe (below) | if DC and THz disagree by 10×, the inversion — not the technique — is the problem |

---

## 9. Proposed first steps

1. **Four-point-probe the actual paper.** Literature SWCNT films run 452–1938 S/cm in-plane;
   we extract 52–62 S/cm. Porosity plausibly explains a large dilution factor, but this is a
   half-day known-answer test and everything downstream is conditioned on it.
2. **Prototype in the existing 45° geometry.** Two magnet settings × two GaP azimuths + gold
   mirror; validate the eigenvalue calibration on HR-Si, which must return $n = 3.418$, flat.
   No mechanical rebuild. This tests the whole measurement and analysis chain, and directly
   answers whether the ~10% systematic is common-mode — the central claim of this report.
   *Expect no conditioning improvement at 45°; that is not what this step tests.*
3. **Adopt the acquisition protocol from day one** (§4.3), because it costs nothing and the
   alternative is the worst option available: 6–12 emitter settings rather than 2, interleaved
   rather than sequential, with a one-parameter drift ramp fitted and the harmonic residual
   logged as a run-time quality flag.
4. **Then decide on the 70–75° rebuild**, informed by (2).
5. In parallel: flat rigid backing for the paper, and a tilt-measurement procedure good to 0.05°.

---

### Appendix — reproducing the numbers

| claim | source |
|---|---|
| Conditioning vs angle, error channels, detection geometry, eigenvalue calibration, error budget | `explorations/thz_ellipsometry/ellipsometry_conditioning_analysis.py` |
| Anisotropic Jones matrix, cross-polarization vs azimuth, identifiability, scheme comparison | `explorations/thz_ellipsometry/anisotropic_reflection.py` |
| Drude band / Fisher analysis | `explorations/thz_ellipsometry/drude_band_study.py` |
| Mount-to-mount penalty | `explorations/thz_ellipsometry/mount_penalty_study.py` |
| Acquisition protocol: over-determination, drift and ordering | `explorations/thz_ellipsometry/harmonic_overdetermination.py` |

All three modules carry known-answer validation (Fresnel round-trip, common-mode cancellation,
perfect-conductor limit, isotropic and reciprocity checks on the anisotropic solver) that runs
before any result is printed.
