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

### The three calibration steps, and why they are not circular

1. **Gold → channel ratio.** $\rho_{\text{gold}} = -1 + (2/N_{\text{gold}})\sin^2\theta/\cos\theta$
   is known from Fresnel to ~0.1% because $|N_{\text{gold}}| \sim 900$ at 1 THz. One complex,
   frequency-flat number comes out. *Its measured frequency-flatness is itself a diagnostic* —
   structure in it means something else is wrong.
2. **HR-Si → incidence angle.** Hold $n = 3.4175$, $k = 0$ fixed and fit the single parameter θ.
   Expected precision ±0.01° (F36).
3. **Doped Si → the actual test.** Full inversion with the channel ratio and θ both fixed from
   (1) and (2). Compare σ_dc against a four-point probe or the supplier specification.

Step (2) consumes HR-Si as a calibrator, so "HR-Si returns 3.418" is **not** an independent
result. Two genuine tests survive it: the *residual* of the one-parameter θ fit across the whole
band (does a single angle explain every frequency?), and the recovered $k_{\text{Si}}$, which was
never fitted and must come out at zero. The independent validation is step (3).

A config switch supports the alternative: set θ mechanically, use HR-Si purely as a validation
sample, and accept worse angle knowledge.

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
| V6 | **Real HR-Si**: one-parameter θ fit residual, and $k$ | residual flat across band; $\|k\| < 0.05$ |
| V7 | **Real doped Si**: σ_dc against four-point probe | within 15%, and the Drude lineshape fits with rms < 0.03 |
| V8 | Repeatability: remount and remeasure | ρ reproducible to < 1% |

V7 is the result that decides whether the method works on our bench. V8 is the one that decides
whether it solves the problem we actually have, since remount reproducibility is what defeated
the CNT campaign.

**Operating protocol baked into the driver** (F35, F37):
- interleave the polarization settings, never run them sequentially;
- 6–12 emitter angles over 180°, not 2;
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

## 9. Open questions for the bench

None of these block steps 0–2; all are needed for step 3.

1. **How will the acquisition software record the polarization angle?** The plan assumes a
   `key=value` token (`pol=15`) in the filename, which the existing grammar already extracts.
   If it will be something else, `loader.py` needs one more adapter and it is better to know now.
2. **Do we already have HR-Si and doped-Si acquisitions at several polarization angles**, or does
   step 3 need a fresh session? If fresh, it should follow the protocol in §6 from the start.
3. **Is there a four-point probe number for the doped wafer**, or a supplier resistivity? V7 needs
   an independent truth.
4. **What is the GaP crystal's current azimuth relative to the probe polarization?** Phase 1 works
   at any azimuth except the two degenerate ones, but near the balanced 31.72° the two channels
   have equal sensitivity and the conditioning is best.
5. **Measured angular spread at the sample plane** (OQ10/OQ12), if the blur correction is to be
   enabled at all.

---

## 10. Deliberate non-goals for Phase 1

- No anisotropy, no cross-polarization, no Mueller formalism.
- No air-gap or window geometry — bare reflection only.
- No CNT measurements.
- No change to `thz_adapter` or the existing run_me scripts.
- No migration of anything into `thz-core` until the package has earned it on real data.
