# THz ellipsometry MVP — implementation plan

*2026-10-01. Plan for a discrete, self-contained ellipsometry package validated on HR-Si and
doped Si. Physics and design rationale live in `reports/thz_ellipsometry_prospecting_report.md`
and lab notebook F33–F37; this document is the build spec.*

---

## 1. Scope

**Build the minimum code that can answer one question: does ellipsometry, on our bench, return
the known optical constants of silicon?**

Everything else — CNT paper, anisotropy, air gaps — is downstream of that answer and is
deliberately out of scope here.

**Phase 1 (this plan).** Isotropic samples. The THz polarization is rotated at the emitter
(magnet only); the GaP crystal is **not** moved. One detection projection.

> **Confirming the premise: yes, this is correct.** With a fixed detection azimuth, a full
> emitter-polarization series yields exactly two complex numbers per frequency — rank 2 of the
> 4-element Jones matrix (F34). That is *complete* for an isotropic sample and rank-deficient
> only for an anisotropic one. Silicon, doped or not, is cubic and therefore optically isotropic,
> so Phase 1 is exactly the right scope for the validation case.

**Phase 2 (later, not planned in detail here).** Second GaP azimuth → full Jones matrix →
generalized ellipsometry, eigenvalue calibration, anisotropy, model-free tilt readout.

### What Phase 1 cannot do, stated up front

- **It cannot see out-of-plane sample tilt.** That lives in $r_{ps}$, which needs two detection
  azimuths. In Phase 1 a tilt biases ρ silently. Size it: 0.1° of tilt gives
  $|r_{ps}/r_{ss}| \approx 0.003$, so roughly 0.3% in ρ and Δ|N| ≈ 0.05 for a CNT-like index —
  far less for silicon. Acceptable for the MVP, handled by the timing-null procedure (F37).
- **It cannot do the eigenvalue calibration** (needs a 2×2 measured block). Phase 1 uses the
  simpler scalar channel-ratio calibration below, which is exact for an isotropic sample.

---

## 2. Package layout

A discrete package at the repo root, deliberately separate from the 7800-line `thz_adapter`:

```
ellipsometry/
    __init__.py            public surface: a dozen names, nothing else
    model.py               forward model (pure)
    harmonic.py            emitter-angle harmonic fit + drift ramp + residual flag (pure)
    calibration.py         channel ratio from gold; incidence angle from silicon (pure)
    inversion.py           rho -> n,k, with optional angular-blur forward model (pure)
    materials.py           HR-Si, doped-Si Drude, gold reference indices (pure)
    simulate.py            synthetic dataset generator (pure)
    validation.py          known-answer checks with explicit pass criteria
    stages.py              @ellipsometry_stage registry + DataSet-aware wrappers
    loader.py              grouping by polarization token, reference pairing
    report.py              text report + figures
run_me_ellipsometry.py     config dict + driver, in the house style
tests/
    test_ellipsometry_model.py
    test_ellipsometry_harmonic.py
    test_ellipsometry_calibration.py
    test_ellipsometry_inversion.py
    test_ellipsometry_workflow.py      headless end-to-end on synthetic data
```

**Everything above `stages.py` is pure**: no `DataSet`, no file I/O, no global state, explicit
inputs and outputs. `stages.py` and `loader.py` are the only files that know this repo exists.
That boundary is what makes the package liftable into `thz-core` later if it earns it.

### Reuse versus build

| Reuse from the repo | Build new in `ellipsometry/` |
|---|---|
| `.acc` loading, grouping, filename grammar (`DataSet`, `FilenameInfo`) | the emitter-angle harmonic fit |
| preprocessing stages: baseline, taper, fixed-width window, FFT | channel-ratio calibration |
| `thz_core.invert` Fresnel primitives and p-pol branch vote | incidence-angle fit from silicon |
| noise model / repeat-scan uncertainty | angular-blur forward model |
| `display.py` plotting registry | the synthetic generator |
| `session_bundle` / `pipeline_registry` for replayable runs | validation criteria |

Nothing is reimplemented that already exists and is tested. The `explorations/thz_ellipsometry/`
modules are **prototypes, not dependencies** — the production code re-derives what it needs with
tests, and the explorations stay as the record of why.

### One-place registration

`pipeline_registry.CORE_STAGE_NAMES` is a hand-maintained tuple, so adding a stage there means
editing two places. We will not propagate that pattern. `ellipsometry/stages.py` uses a
decorator:

```python
@ellipsometry_stage          # registers for dispatch, replay, AND the help listing
def fit_emitter_harmonic(dataset, *, config=None): ...
```

and exposes `ELLIPSOMETRY_STAGES` as the single source of truth. `stages.py` then publishes that
set *into* `pipeline_registry` at import time, so bundles and replay keep working without the
registry growing a second hand-edited list. Retrofitting the existing monolith is out of scope.

---

## 3. What the code actually computes

### Measurement model

For emitter polarization angle α and frequency ω, with everything unknown folded into two
complex channel constants:

$$S(\alpha, \omega) = \big[\,P(\omega)\cos\alpha + Q(\omega)\sin\alpha\,\big]\;e^{i\omega\,\delta t(k)}$$

where $P = d_p\,r_p\,E_0$ and $Q = d_s\,r_s\,E_0$, and $\delta t(k)$ is the drift of the $k$-th
acquisition. The emitted spectrum $E_0$, the beam path, the detector gain and any common timing
all cancel when $P$ and $Q$ are divided.

### Data flow

```
 .acc files
     |  loader.py        group by pol=<deg>; pair sample <-> gold reference
     v
 per-angle traces
     |  existing stages  baseline -> fixed-width window -> FFT
     v
 per-angle spectra  S(alpha_k, omega)
     |  harmonic.py      least squares for P, Q  (+ one-parameter drift ramp, F35)
     |                   residual norm -> quality flag
     v
 P(omega), Q(omega)        ... and the same for the gold reference
     |  calibration.py    channel ratio  d_p/d_s = (P_gold/Q_gold) / rho_gold
     v
 rho(omega) = (P/Q) * (d_s/d_p)
     |  calibration.py    incidence angle theta from the HR-Si measurement
     |  inversion.py      rho -> n, k   (+ optional angular-blur forward model)
     v
 n(omega), k(omega), sigma(omega)   -> report.py, session bundle
```

### The calibration chain

*(Revised 2026-10-01 after Samuel pointed out the flaw in the first version: both silicon wafers
are polished and reflect visible light perfectly well, so the "no visible alignment handle"
problem is a CNT-paper problem, not a silicon problem. The incidence angle can therefore be set
mechanically for this test, and HR-Si is freed to be what it should be — a validation sample.)*

1. **Gold → channel ratio $d_p/d_s$.** The one thing mechanical alignment cannot give us.
   $\rho_{\text{gold}}$ is known from Fresnel to ~0.1% because $|N_{\text{gold}}| \sim 900$ at
   1 THz, so one complex, frequency-flat number comes out. Swapping the gold in is **safe**,
   because what we extract from it is a polarization *ratio*, which is itself immune to the
   placement error that poisons an amplitude reference. Its measured frequency-flatness is a
   diagnostic in its own right.
2. **Incidence angle θ → mechanical**, aligned with a visible laser off the polished wafer and
   confirmed by the timing-null procedure (F37). 0.1° is enough (it costs Δn ≈ 0.02 on silicon),
   and visible autocollimation does far better than that.
3. **HR-Si → validation.** Expect $n = 3.4175$ flat, and $k$ consistent with zero.
4. **Doped Si → validation.** Against **transmission on the same wafer**, which is the house
   standard and better than any datasheet.

**Cross-check, not calibration:** also fit θ from the HR-Si measurement as a free parameter and
compare it to the mechanical value. Agreement validates both; disagreement localises the problem
to the geometry rather than the analysis. This is strictly better than consuming HR-Si as a
calibrator, which is what the first draft of this plan proposed.

### Why HR-Si *and* doped Si — the k question

Samuel's reason for wanting both: we have previously had poor sensitivity to $k$ in silicon and
it was unclear whether that was physical, instrumental or analytical. The pair settles it,
because one wafer has essentially no absorption and the other has plenty.
Computed in `explorations/thz_ellipsometry/silicon_k_sensitivity.py`:

**What k actually is at 1 THz:**

| sample | k |
|---|---|
| HR float-zone Si (α = 0.05 /cm) | 0.00012 |
| HR Si, pessimistic (α = 0.3 /cm) | 0.00072 |
| doped 10 Ω·cm | 0.0098 |
| doped 1 Ω·cm | 0.102 |
| doped 0.1 Ω·cm | 1.47 |

**What we can measure**, with a 0.5% additive field-noise floor (noise enters on the two
*fields*, not on ρ — modelling it as a relative error on ρ is wrong near Brewster, where
$r_p \to 0$):

| θ | δk, HR-Si | δk, doped 1 Ω·cm |
|---|---|---|
| 45° | 0.057 | 0.053 |
| 65° | 0.034 | 0.032 |
| 70° | 0.032 | 0.031 |
| 73.7° (Brewster) | 0.031 | 0.030 |
| 80° | 0.034 | 0.033 |

**Three results, and the third is the useful one:**

- **δk is essentially flat with angle** — a factor of 1.8 from 45° to 73°, and no Brewster
  miracle. At Brewster the p channel vanishes and the fixed field noise dominates completely, so
  the sensitivity gain and the signal collapse cancel. The angle should be chosen for the other
  reasons (blur, conditioning on n), not for k.
- **δk ≈ δn ≈ 0.03** at every angle, to within 1%. The measurement has roughly *isotropic*
  precision in the complex index plane. **So k is not intrinsically harder to measure than n —
  it is simply a smaller number.** For n = 3.4 that floor is 1% relative; for k = 0.1 it is 30%.
- **Therefore: for HR-Si the poor k sensitivity is PHYSICAL.** k = 1.2 × 10⁻⁴ sits ~250× below
  the floor. No reflection measurement at any angle with any realistic noise will see it, and
  recovering "k = 0 ± 0.03" on HR-Si is the *correct* answer, not a failure.

**Consequence for sample choice — worth acting on before the session.** The floor scales linearly
with field noise, so the verdict depends entirely on the doping:

| sample | k | SNR on k | verdict |
|---|---|---|---|
| HR-Si | 0.00012 | 0.00 | invisible |
| doped 10 Ω·cm | 0.0098 | 0.31 | invisible |
| doped 1 Ω·cm | 0.102 | 3.4 | measurable |
| doped 0.1 Ω·cm | 1.47 | 74 | easy |

**Use the most heavily doped wafer available, ideally ≤ 1 Ω·cm.** A 10 Ω·cm wafer would leave the
k half of the test inconclusive — n and the transmission comparison would still work, but the
question Samuel actually wants answered would not get an answer.

---

## 4. Module specifications

**`model.py`** — `jones_isotropic(index, incidence_angle, index_incident)`,
`detection_vector(probe_azimuth)`, `emitter_field(alpha)`, `measured_amplitude(...)`. One
forward-model function, used by the simulator, the fitter and the validator alike, so there is
exactly one definition of the physics.

**`harmonic.py`** — `fit_emitter_harmonic(angles, spectra, frequencies, *, drift_model)` returning
`HarmonicFit(channel_p, channel_s, drift, residual_norm, degrees_of_freedom)`.
`drift_model` is injected: `"none" | "linear_ramp" | "per_acquisition"`. Default `"linear_ramp"`
(F35: flat 0.17 index error from 0 to 200 fs of drift, with no penalty at zero drift).
`harmonic_residual_norm()` is the run-time quality flag.

**`calibration.py`** — `channel_ratio_from_reference(fit_reference, reference_index, angle)`,
`incidence_angle_from_silicon(fit, frequencies, *, bounds)`, and
`ChannelCalibration` carrying both plus their uncertainties. All pure functions of arrays.

**`inversion.py`** — `index_from_ellipsometric_ratio(rho, angle, index_incident)` (closed form),
plus `index_from_ellipsometric_ratio_with_blur(rho, angle, angular_spread)` which inverts the
`<r_p>/<r_s>` angular average numerically (F36/F37). Blur correction is **opt-in and requires a
measured spread** — a 50%-wrong correction is worse than none.

**`materials.py`** — `HIGH_RESISTIVITY_SILICON = 3.4175 - 0j`, `gold_index(frequency)`,
`doped_silicon_drude(frequency, carrier_density, mobility)`. Literature values with sources in
docstrings, so the validation targets are not scattered through the code.

**`simulate.py`** — `synthesize_measurement(...)` producing the same arrays the real loader
produces, with injectable noise, per-acquisition drift, sample tilt, angular blur and channel
imbalance. **This is a first-class deliverable, not a test fixture**: it is how the pipeline is
debugged without beam time, and how every failure mode gets a known answer.

**`validation.py`** — `validate_against_silicon(result)` returning a `ValidationReport` with
explicit pass criteria (below), and a `round_trip_known_answer()` that drives `simulate.py`
through the whole chain and asserts recovery.

**`stages.py` / `loader.py` / `report.py`** — the thin adapter layer described above.

---

## 5. Config schema (`run_me_ellipsometry.py`)

House style, explicit, no hidden defaults:

```python
config = {
    "general": {"show_graph": True, "save_session": True, "session_notes": "..."},
    "geometry": {
        "incidence_angle_deg": 70.0,        # nominal; refined by the silicon fit
        "incidence_angle_source": "silicon", # or "mechanical"
        "index_incident": 1.0,
    },
    "polarization": {
        "angle_token": "pol",               # filename key=value token, e.g. pol=15
        "angles_deg": None,                 # None = take whatever is on disk
        "probe_azimuth_deg": 31.72,         # GaP [001] vs probe; balanced detection
        "drift_model": "linear_ramp",
    },
    "reference": {"vocabulary": ["gold", "mirror"], "index": "gold"},
    "blur": {"enabled": False, "angular_spread_deg": None},  # requires a knife-edge number
    "band": {"frequency_min_thz": 0.8, "frequency_max_thz": 3.0},
    "validation": {"expect": "hr_silicon", "tolerance_n": 0.02, "tolerance_k": 0.05},
}
```

---

## 6. Validation plan — what "done" means

Run in this order; each gate must pass before the next is meaningful.

| # | Test | Pass criterion |
|---|---|---|
| V1 | Forward/inverse round trip on noiseless synthetic data | ρ and n,k recovered to < 1e-9 |
| V2 | Harmonic fit recovers planted P, Q under noise + drift | unbiased; residual flag fires when drift is unmodelled |
| V3 | Channel-ratio calibration on synthetic gold | recovers the planted $d_p/d_s$ to < 0.5% |
| V4 | Angle fit on synthetic HR-Si | recovers planted θ to < 0.05° |
| V5 | Full synthetic chain, realistic noise/drift/blur | $\|n - 3.4175\| < 0.02$, $\|k\| < 0.05$ over 1–2 THz |
| V6 | **Real HR-Si**: n against transmission on the same wafer | $\|n - 3.4175\| < 0.02$; $k$ consistent with 0 within the ~0.03 floor; fitted θ agrees with the mechanical setting to 0.1° |
| V7 | **Real doped Si**: n and k against **transmission on the same wafer** | n within 0.02, k within 0.03 (absolute, not relative), Drude σ_dc within 15% |
| V8 | Repeatability: remount and remeasure | ρ reproducible to < 1% |

**Transmission on the same wafers, measured in the same session, is the truth standard** — better than any datasheet, and it uses a pipeline we already trust. V7 is the result that decides whether the method works on our bench. V8 is the one that decides
whether it solves the problem we actually have, since remount reproducibility is what defeated
the CNT campaign.

> **How many emitter angles? Four. (Revised 2026-10-01 - an earlier draft said 6-12.)**
>
> Two angles *do* measure rho correctly: P and Q are two complex unknowns per frequency and
> alpha = 0, 90 gives two complex equations with a perfectly conditioned design matrix. The
> problem is not the measurement, it is that two angles can neither correct nor even detect a
> drift between them, and the drift nuisance needs at least three angles to be identifiable.
>
> Index error at 1 THz / 70 deg, **48 acquisitions however they are split** (equal total
> measurement time), with sample and reference drifting independently:
>
> | drift rms between settings | 2 angles | 4 angles + drift ramp |
> |---|---|---|
> | 0 fs | 0.040 | 0.042 |
> | 15 fs | 0.084 | 0.042 |
> | 30 fs | 0.145 | 0.041 |
> | 60 fs | 0.278 | 0.041 |
>
> Four angles are **flat** against drift and cost nothing at zero drift. Going past four does
> not help (6, 8 and 12 all land at 0.047-0.051, slightly worse, because at fixed total time
> each angle gets fewer scans). **Four is the smallest number that makes the drift nuisance
> identifiable and still leaves a residual degree of freedom for the quality flag.**
>
> The first-order lever is not the angle count, it is total integration:
>
> | total acquisitions per sample | scans per angle | index error |
> |---|---|---|
> | 16 | 4 | 0.071 |
> | 48 | 12 | 0.042 |
> | 96 | 24 | 0.030 |
> | 192 | 48 | 0.021 |
>
> (single frequency at 1 THz; the validation uses the band median, which does better.)

**Operating protocol baked into the driver** (F35, F37):
- interleave the polarization settings, never run them sequentially;
- **4 emitter angles** (0, 45, 90, 135) over 180 degrees, not 2 - see the box below;
- gold reference the **same size as the sample, in the same mount**;
- focus the full beam rather than clipping it;
- find the tilt null by minimising pulse arrival time (fit $a + bt + ct^2$, take the vertex).

---

## 7. Testing strategy

Per the house rule, two kinds, both required before a phase is called done.

**Unit tests** — one file per pure module, known-answer where possible: Fresnel round trip,
harmonic fit against a planted sinusoid, drift separability, channel-ratio recovery, blur
averaging against the plane-wave limit, silicon angle fit.

**Headless workflow test** — `test_ellipsometry_workflow.py` drives `run_me_ellipsometry.py`'s
pipeline end to end on a synthesized dataset written to a temporary directory in real `.acc`
format, through the real loader, with `show_graph=False`, and asserts V5. No hardware, no GUI,
no network. This is the test that catches integration rot.

Target: the existing suite stays at 370 passed, plus roughly 35–45 new tests.

---

## 8. Build order

> **Status 2026-10-01: steps 0, 1 and 2 are DONE.** `ellipsometry/` is implemented and
> `run_me_ellipsometry.py` runs the whole chain against synthetic `.acc` files on disk.
> **79 new tests (449 total, up from 370).** Gates V1-V5 pass. Step 3 needs the bench.
>
> One design correction made during the build, worth recording because an earlier draft of this
> plan got it wrong: **a constant emitter-angle offset is NOT absorbed by the gold calibration.**
> If the emitter's angular zero is offset from the plane of incidence by δ, the measured channel
> ratio is a Möbius transform of ρ, not a scaled copy:
>
> $$m = \frac{C\rho + t}{1 - C t \rho}, \qquad C = d_p/d_s,\; t = \tan\delta$$
>
> because the commanded "p" setting then contains sin δ of s. Gold alone cannot separate C from
> t — its ρ is essentially −1 at every frequency, so it gives one complex constraint for three
> real unknowns. Consequences, all now implemented:
> - **Specification:** leaving the offset uncorrected costs |ΔN| ≈ 0.009 per 0.1° at 70°
>   incidence and 0.002 per 0.1° at 45°, essentially independent of the sample. **Keep the
>   emitter zero within ~0.2° of the plane of incidence** and it stays inside the 0.02 tolerance.
> - **If it is known**, pass `geometry.emitter_offset_deg` and it is removed exactly.
> - **If it is not**, `fit_instrument_from_references` fits (C, t) jointly from two references
>   whose ρ differ — but that consumes both as calibrators.
> - **Best:** measure it once during setup. A wire grid aligned to the plane of incidence gives
>   a sharp, sign-sensitive null in coherent detection, and it never has to be in the beam again.


| Step | Deliverable | Depends on data? |
|---|---|---|
| 0 | Package skeleton, `model.py`, `materials.py`, unit tests | no |
| 1 | `simulate.py` + `harmonic.py` + `calibration.py` + `inversion.py`, V1–V4 | no |
| 2 | `loader.py`, `stages.py`, `run_me_ellipsometry.py`, workflow test, V5 | no (synthetic `.acc`) |
| 3 | Run on real HR-Si and doped Si: V6, V7, V8 | **yes** |
| 4 | `report.py` figures + session bundle integration | no |
| 5 | *(Phase 2)* second GaP azimuth, eigenvalue calibration, anisotropy | yes |

Steps 0–2 need no beam time, which is the point: the whole chain can be built and validated
against known answers before a single silicon wafer is measured, and then step 3 is a short
session rather than a debugging marathon.

---

## 9. Bench questions — answered 2026-10-01

1. **Polarization token** — defaulting to the `pol=15` `key=value` form, which the existing
   filename grammar already extracts regardless of position. Untested in anger, so `loader.py`
   reports clearly what it found and what it expected.
2. **Data** — step 3 is a fresh session, so the protocol in §6 applies from the first scan.
3. **Truth standard** — **transmission on the same wafers**, not datasheet values.
4. **GaP azimuth** — currently at one of the two *degenerate* orientations (parallel or
   perpendicular to the gate polarization), where one channel is blind and ρ cannot be measured
   at all. **Moving it to ~31.7° is a prerequisite, not an optimisation.** Worth recording which
   degenerate orientation it starts from, since that fixes the sign convention for the rotation.
5. **Angular spread** — unmeasured; the blur correction stays disabled (`blur.enabled = False`)
   until there is a knife-edge number. A wrong correction is worse than none.

**Open and worth settling before the session:** what is the resistivity of the doped wafer? If it
is 10 Ω·cm or higher, k will sit below the measurement floor and that half of the validation will
be inconclusive.

## 10. Deliberate non-goals for Phase 1

- No anisotropy, no cross-polarization, no Mueller formalism.
- No air-gap or window geometry — bare reflection only.
- No CNT measurements.
- No change to `thz_adapter` or the existing run_me scripts.
- No migration of anything into `thz-core` until the package has earned it on real data.
