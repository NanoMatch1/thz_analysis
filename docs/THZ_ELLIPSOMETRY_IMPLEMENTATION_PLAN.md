# THz Ellipsometry — Implementation Plan

**Source physics/measurement plan (authoritative):**
`explorations/thz_ellipsometry/thz_tds_ellipsometry_self_referencing_plan.md` (referred to below as
"the plan", with its section numbers §N). It supersedes the scope of `docs/ELLIPSOMETRY_MVP_PLAN.md`,
which remains the record of how Phase 1 was validated (gates V1–V8) and is not deleted.

**Prototypes, not dependencies:** the other modules in `explorations/thz_ellipsometry/` are lifted
from (with tests) where useful, never imported.

---

## 0. Decisions in one screen

1. **Evolve the existing package; do not start a parallel one.** `ellipsometry/` (commit `d5d1652`,
   79 tests) already implements most of the plan's *isotropic mode* (§6) as a pure-core / thin-adapter
   package with a stage registry. Writing a second module would duplicate the Fresnel model, harmonic
   fit, calibration, inversion and simulator. Instead: **rename `ellipsometry/` → `thz_ellipsometry/`**
   and extend it.
2. **One run script:** `thz_ellipsometry_run_me.py` (replaces `run_me_ellipsometry.py`), module-level
   `config` dict like the other run_me scripts, `--simulate <sample>` and `--no-graph` flags kept.
   `config["mode"]` selects `"isotropic"` or `"generalized"` (GE).
3. **The layer rule stays the rule.** `thz_ellipsometry/core/` is pure (arrays in, arrays out, no
   DataSet, no I/O, no globals except registries) and liftable into `thz-core`. `thz_ellipsometry/adapters/`
   is the only part that knows the repo exists, and it is where repo layers are reused.
4. **Registries, one place each:** stages (exists), acquisition modes, calibration sources, inversion
   routes. Each new route/source/mode is added by defining it — nothing else to edit. Diagnostics use
   the repo's existing `@diagnostic` registry so they land in `docs/assumptions_ledger.md` automatically.
5. **Convention:** keep the repo's `N = n − i k`, `k ≥ 0` (`thz_core` Fresnel). Verified: `thz_core`
   gives `ρ_Si(45°) = −0.6509` and `ρ_gold ≈ −1`, i.e. the plan's §6.1 sign convention for ρ already
   holds; only the `n ± ik` sign differs, which the plan (§6.1) explicitly allows as long as it is
   uniform. The §6.5 "sign-flipped silicon reads ≈ 0.52" signature becomes a registered diagnostic.

---

## 1. Gap analysis: the plan vs. what exists

| Plan section | Feature | Status in `ellipsometry/` | Work |
|---|---|---|---|
| §3.1, §6.3 | Ratio R = S_p/S_s, κ calibration | Done (harmonic fit `P/Q`, `channel_ratio_from_reference`) | Rename κ ↔ `channel_ratio` consistently |
| §4.1–4.2 | Magnet states 0/90/180/270, **±M background differencing** | Partial: harmonic fit has no background term; angles are "THz pol", not magnet | Add background column + magnet→polarization mapping |
| §4.2 | Interleaved cycles, ratio-of-averages | Partial: one averaged spectrum per angle; drift ramp is over *angle order* | Fit per acquisition with real timestamps |
| §4.4 | Magnet-zero offset δ | Done (Möbius treatment, `emitter_offset_deg`) — and more correct than the plan's wording | Keep; feed from wire-grid null value |
| §5.1 | Gold → κ | Done | — |
| §5.2 | HR-Si → θ_eff | Done as *cross-check* (`fit_incidence_angle`) | Make it a selectable calibration source |
| §5.3 | Tilt test (θ_eff vs in-plane tilt) | Missing | Multi-dataset stage: θ fit per tilt, report slope |
| §6.2 | Common window, band by SNR | Done (common-centre Hann, amplitude floor) | Swap to repo preprocess + noise-model SNR |
| §6.4 | Closed-form isotropic inversion | Done | — |
| §6.5 | Si flat 11.68, branch check, sign-flip signature | Partial (V-gates) | Register as `@diagnostic`s |
| §7.2 | Aligned-axis anisotropic closed forms; aligned-azimuth pair route | Missing | New inversion route |
| §7.3 | EO detection vector `(sin2φ, 2cos2φ)` | Done (`electro_optic_detection_vector`) | Generalize to detection *matrix* D |
| §7.4 | M = D⁻¹S, normalized Jones matrix | Missing | New `jones.py` |
| §7.5 | Magnet-sweep calibration of D, E_p/E_s on gold | Missing | New calibration source |
| §7.6 | Berreman 4×4 semi-infinite, ε_o/ε_e fit, ψ global | Prototype only (`anisotropic_reflection.py`, validated vs Fresnel) | Lift into core with tests |
| §7.7 | GE cross-validation diagnostics | Missing | Diagnostics |
| §7.8 | HWP wedge / probe-walk test on gold | Missing | Diagnostic stage (90°- vs 360°-periodicity) |
| §7.9, §7.2 | Out-of-plane: ε_⊥ from ρ + ε_∥ (transmission) | Missing | Inversion route that consumes a transmission bundle |
| §8.1–8.2 | Angle affine error, conditioning ∂ε/∂ρ | Missing | `sensitivity.py`, plotted with results |
| — | Probe-rotation self-calibration (`c4817b6`) | Prototype only | Calibration source (not in the plan, see §2) |
| — | Out-of-plane tilt as fitted nuisance (`582b43e`) | Prototype only | Option in isotropic inversion (see §2) |

---

## 2. Where the plan and our established findings disagree (resolved in code, not silently)

These are the points where implementing the plan literally would reintroduce errors we have already
found. Each becomes either a config switch or a diagnostic.

1. **§5.1 "placement does not matter" is true for scalars, false for rotations.** Height, gain,
   roughness, truncation multiply both polarizations and cancel. *Out-of-plane tilt rotates the p/s
   frame*; it does not cancel against gold. Established (`582b43e`): for a **dispersive** sample the
   tilt is fittable from the sample's own data (ψ enters as `(ρ−1)ψ`, frequency-dependent; κ is flat).
   → `inversion.fit_out_of_plane_tilt: bool` (default **on** for dispersive samples, off for HR-Si where
   it is degenerate with κ). Report fitted tilt ± error.
2. **§5.2 HR-Si as the θ calibrator vs. as validation.** For the silicon validation run, θ is set
   mechanically and HR-Si is a *test*; consuming it as the calibrator makes "Si returns 3.418"
   circular. For unknown samples later (CNT), HR-Si → θ_eff is exactly right. → calibration source
   `incidence_angle.source: "mechanical" | "fit_from_reference"`; the run report states which.
3. **§4.4 / §8.3 magnet offset.** The plan says "few % per degree, does not cancel" — correct, and
   the exact form is a Möbius map `m = (Cρ + t)/(1 − Ctρ)`, already implemented. Gold alone cannot
   separate C from t; keep the wire-grid value as config and the two-reference fit as fallback.
4. **§4.1 magnet angle ≠ THz polarization angle.** THz polarization ⟂ **M**. Filenames record the
   magnet state; config carries `magnet_to_p_offset_deg` (90 nominal, refined by the §4.4 wire-grid
   null), so "magnet 0° gives p" is a calibration value, not an assumption baked into code.
5. **§6.5 Abelès check (`r_p = r_s²` at 45°) cannot be tested from ρ alone** — it is an identity of
   the isotropic model, so any isotropic inversion satisfies it. It is only a check when compared with
   an *absolute* r_s. → implement as an optional cross-check against the existing gold-referenced
   single-reflection pipeline (`run_me_reflection_single.py`): ellipsometry predicts r_s = ρ; the
   mirror-referenced measurement must agree within its known misalignment error.
6. **Probe-rotation self-calibration is newer than the plan.** Ratioing the same sample at two probe
   azimuths cancels the sample and yields κ (up to one crystal-orientation number χ) with nothing in
   the THz path moving and no gold. GE mode already acquires two probe settings, so this comes almost
   free → calibration source `probe_rotation`; gold κ and probe-rotation κ cross-check each other.
7. **§4.2 angles.** The MVP default was 0/45/90/135 THz-pol; the plan specifies magnet 0/90/180/270.
   Both are supported by the same general fit (§3.2 below). 0/90/180/270 is the default because the
   ±M pairs remove non-magnetic background physically, and four acquisitions still leave the drift ramp
   identifiable (F-finding behind `bfd0a72`: four angles, not more, at fixed total time).

---

## 3. Architecture

### 3.1 Package layout

```
thz_ellipsometry/
  __init__.py                 public API re-exports + the layer rule in the docstring
  core/                       PURE
    conventions.py            N = n - ik; FFT-sign helpers; one place to read the convention
    model.py                  isotropic Fresnel (via thz_core), rho, psi/Delta       [exists]
    detection.py              EO detection vector d(phi), detection matrix D(phi_k; gamma, phi0),
                              degeneracy checks                                      [split from model.py]
    acquisition_model.py      S_k = c * d_k^T J R(beta_k) E + B_k, exp(i w tau(t_k)):
                              magnet->polarization map, background term, per-acquisition rows
    harmonic.py               variable-projection fit of the model above              [generalize]
    calibration.py            kappa (gold), theta fit, Moebius emitter offset          [exists]
    calibration_sources.py    @calibration_source registry: gold_reference, probe_rotation,
                              magnet_sweep_gold (D + Ep/Es), stored
    jones.py                  M = D^-1 S, normalized Jones ratios (rpp/rss, rps/rss, rsp/rss)
    anisotropic.py            Berreman 4x4 semi-infinite J(eps_tensor, theta); tensor rotation;
                              §7.2 aligned closed forms                              [lift from prototype]
    inversion.py              isotropic closed form, blur-aware, tilt nuisance         [exists + tilt]
    inversion_routes.py       @inversion_route registry (§3.3)
    sensitivity.py            §8.1 affine angle map, §8.2 d eps/d rho, uncertainty propagation
    materials.py              Si, doped Si, gold, uniaxial test materials             [exists]
    simulate.py               synthetic spectra + .acc files for EVERY mode/route     [extend]
    validation.py             known-answer gates                                       [exists]
  adapters/                   knows the repo
    loader.py                 .acc -> per-acquisition series via repo readers + filename grammar
    preprocess.py             thin wrapper on thz_core/preprocess (baseline, taper, fixed window)
    noise.py                  thz_core.noise per magnet state -> per-bin sigma -> fit weights + errors
    diagnostics.py            @diagnostic registrations (into the repo's ledger)
    stages.py                 @ellipsometry_stage registry + run_ellipsometry(config)  [exists]
    acquisition_modes.py      @acquisition_mode registry: isotropic, generalized
    export.py                 result bundle (recipe/config + arrays + report.md), quantity
                              registration for display/results viewer
    report.py                 [method_name] outcome lines + figures                   [exists]
thz_ellipsometry_run_me.py
tests/test_thz_ellipsometry_core_*.py       unit, per core module
tests/test_thz_ellipsometry_workflow_*.py   headless end-to-end, per mode/route
```

`core/` imports only numpy/scipy/`thz_core`. `adapters/` may import `dataset_core` and `thz_core`.
A test enforces this (import-graph check on `core/`), so the boundary cannot erode.

### 3.2 The one generalized measurement model

Every mode is the same equation with different design rows, so one fitter serves all:

```
S_k(w) = c(w) * [ d(phi_k)^T  J(w)  P(beta_k - beta_0)  E(w) ] * exp(i w tau(t_k))  +  B(phi_k, w)
```

- `beta_k` magnet angle of acquisition k, `phi_k` probe setting, `t_k` timestamp (from the .acc header
  via `acquisition_tracking.AcquisitionSeries.elapsed_seconds`).
- `B` is the non-magnetic background: one complex value per probe setting per frequency; it is what
  ±M differencing removes, done here as a fitted column (identical to differencing for 0/180 pairs,
  but also valid for unpaired angle sets).
- `tau(t)` drift nuisance: `none | linear_ramp | free` — now over **real elapsed time**, so interleaved
  cycling (§4.2) is modelled exactly instead of approximated by angle order.
- Isotropic mode = one `phi`, J diagonal, unknowns `P = c d_p r_p E_p`, `Q = c d_s r_s E_s` (the MVP).
  GE mode = two `phi`, unknowns are the 2×2 block `c D J E`.
- Fitting every acquisition jointly is the least-squares equivalent of the plan's "ratio of averages",
  with noise-model weights.

### 3.3 Registries (single source of truth each)

| Registry | Entries (initial) | Declares |
|---|---|---|
| `@acquisition_mode` | `isotropic`, `generalized` | required filename tokens, required probe settings, which design rows |
| `@calibration_source` | `gold_reference`, `probe_rotation`, `magnet_sweep_gold`, `stored` | what it produces (κ / D / E_p/E_s / θ), what data it needs |
| `@inversion_route` | `isotropic_closed_form`, `aligned_azimuth_pair` (§7.2), `in_plane_uniaxial` (§7.6), `out_of_plane_with_transmission` (§7.2/7.9) | required observables, unknowns, initial-guess strategy, which mode(s) it accepts |
| `@ellipsometry_stage` | load → spectra → fit → calibrate → invert → validate → export | (exists) |
| `@diagnostic` (repo's) | see §5 | stage, assumption, why, remedy |

The run_me's help text and the report's "what ran" section are generated from these registries.
Each route checks at load time that the configured mode supplies its required observables and fails
with a message naming the missing acquisitions, rather than inverting garbage.

### 3.4 What is reused from the repo (and how)

| Repo layer | Used for | Where |
|---|---|---|
| `thz_core` Fresnel (`fresnel_reflection_p/s`) | the isotropic model | `core/model.py` (already) |
| `.acc` reader + `acquisition_tracking.AcquisitionSeries` | per-scan traces **and timestamps** for the drift model; replaces the MVP's private `read_accumulation_file` | `adapters/loader.py` |
| filename grammar (`services/grouping.py`, key=value tokens) | `mag=`, `probe=`, `cyc=` tokens + reference vocabulary | `adapters/loader.py` |
| `thz_core.preprocess` / fixed-width window | baseline, taper, one common window per series (no per-trace re-centring — that would erase the drift being fitted) | `adapters/preprocess.py` |
| `thz_core.noise` (repeat-scan σ_α/β/τ) | per-bin σ per magnet state → weighted fit, error bars on n, k, ε, and the plan's 10× SNR band rule (§6.2) on *measured* noise | `adapters/noise.py` |
| diagnostics `@diagnostic` registry + run-log | every check in §5 | `adapters/diagnostics.py` |
| `quantity_registry` + `display.plot_quantity` | n, k, ε₁, ε₂, σ₁, σ₂, ψ/Δ, Jones ratios plotted in house style with resolution grid | `adapters/export.py`, `report.py` |
| `session_bundle` pattern (recipe.json / arrays / report.md) | replayable result bundle; catalog-indexable | `adapters/export.py` |
| `conductivity_fitting.joint_drude_fit` | Drude fit on doped Si / CNT σ(ω) → resistivity vs four-point probe | post-inversion stage |
| transmission pipeline bundles (`run_me_transmission.py`) | ε_∥ input for the out-of-plane route; transmission truth standard for doped-Si validation | `out_of_plane_with_transmission` route |
| `run_me_reflection_single.py` path | absolute r_s cross-check (item 5 of §2) | optional validation stage |

Not reused: `DataSet`/`thz_adapter` pairing logic (a polarization series is a different shape — one
sample, many states), and the reflection window/air-gap machinery (no window in this geometry).

---

## 4. Filename grammar and config

**Filenames** (key=value tokens, position-free, existing grammar):
```
<sample>_mag=090_probe=31.7_cyc=03.acc        sample, magnet 90 deg, probe 31.7 deg, cycle 3
ref-gold_mag=000_probe=31.7_cyc=01.acc        reference vocabulary marks calibration standards
```
`cyc` is optional (timestamps are authoritative); `probe` is optional in isotropic mode
(defaults to the config value). `mag` replaces the MVP's `pol` token; `pol` is still read, with a
config flag saying whether it means magnet or polarization angle, so the existing synthetic and any
early bench data still load.

**Config sketch** (`thz_ellipsometry_run_me.py`):
```python
config = {
    "mode": "isotropic",                      # registry key: isotropic | generalized
    "data": {"directory": ..., "transmission_bundle": None},
    "geometry": {
        "incidence_angle_deg": 45.0,
        "magnet_to_p_offset_deg": 90.0,       # THz pol is perpendicular to M; refine by wire-grid null
        "emitter_offset_deg": 0.0,            # residual delta (Moebius), from §4.4 null
        "index_incident": 1.0,
    },
    "detection": {
        "probe_settings_deg": [31.72],        # GE: e.g. [0.0, 45.0] (+ more for redundancy)
        "crystal_axis_offset_deg": 0.0,       # gamma; fitted by magnet_sweep_gold / probe_rotation
    },
    "acquisition": {
        "angle_token": "mag", "probe_token": "probe",
        "background_term": True,              # ±M differencing as a fitted column
        "drift_model": "linear_ramp",         # over real elapsed time
    },
    "calibration": {
        "channel": "gold_reference",          # registry: gold_reference | probe_rotation | stored
        "detection_matrix": "magnet_sweep_gold",   # GE only
        "incidence_angle": "mechanical",      # mechanical | fit_from_reference
        "angle_reference": "hr_silicon",
        "stored_path": None,
    },
    "preprocess": {"window_half_width_ps": 3.0, "baseline_fraction": 0.1, "pad_factor": 4},
    "band": {"frequency_min_thz": 0.8, "frequency_max_thz": 3.0, "minimum_snr": 10.0},
    "inversion": {
        "route": "isotropic_closed_form",     # registry
        "fit_out_of_plane_tilt": False,
        "azimuth_deg": None,                  # in-plane uniaxial: None = fit globally
        "blur": {"enabled": False, "angular_spread_deg": None},
    },
    "post": {"drude_fit": False},
    "validation": {"expect": "hr_silicon", "tolerance_n": 0.02, "tolerance_k": 0.05},
    "general": {"show_graph": True, "save_figure": None, "export_bundle": None},
}
```

---

## 5. Diagnostics (registered, never halting — the repo's convention)

Each is an `@diagnostic` with assumption / why / remedy, so the ledger documents itself.

| Diagnostic | Assumption it guards | Source |
|---|---|---|
| `harmonic_residual` | data are a pure first harmonic in magnet angle (+ background) | §4.2, F35 |
| `magnet_reversal_symmetry` | S(β+180) = −S(β) after background; magnet saturates | §4.1 |
| `background_fraction` | non-magnetic background is small vs signal | §4.2 |
| `drift_magnitude` | fitted drift within the range the ramp model handles | F35 |
| `kappa_flatness` | κ is frequency-flat (|κ| and arg κ scatter) | §3.2, §5.1 |
| `probe_degeneracy` | probe setting not within 2° of 0/45 from [001] | §4.3 |
| `silicon_phase_flat_at_pi` | κ-corrected Si ratio has arg ≈ π flat | §5.2 |
| `sign_convention_signature` | Si not inverting to ≈0.52 (ρ→−ρ) | §6.5 |
| `passive_branch` | Re N, k ≥ 0 per bin (convention/calibration error otherwise) | §6.5 |
| `near_mirror_conditioning` | \|∂ε/∂ρ\| below a threshold; flags bins as ρ→−1 | §8.2 |
| `off_diagonal_floor` | isotropic sample in GE gives r_ps, r_sp at the noise floor | §7.7 |
| `probe_walk_signature` | HWP-angle ratios on gold are real & flat (no high-f roll-off / phase ramp) | §7.8 |
| `kappa_sources_agree` | gold κ and probe-rotation κ agree within errors | §2 item 6 |
| `fitted_tilt_magnitude` | fitted out-of-plane tilt plausible (< ~1°) | `582b43e` |

---

## 6. Build order (each phase ends green: unit + headless workflow tests, counts reported)

**Phase 0 — rename and re-seat on repo layers (no behaviour change).**
`git mv ellipsometry thz_ellipsometry`; split into `core/` and `adapters/`; loader moves to
`AcquisitionSeries` + repo grammar; `run_me_ellipsometry.py` → `thz_ellipsometry_run_me.py`; MVP plan
and memory pointers updated. Gate: the 79 existing tests pass unchanged except import paths; the
`--simulate hr_silicon` run gives the same n to 1e-6. Import-boundary test added.

**Phase 1 — the plan's isotropic mode, complete (no beam time).**
Generalized acquisition model (§3.2) with magnet states, background column, timestamp drift;
simulator emits magnet-state files with background, interleaved cycles and drift; noise-model
weighting and error bars; calibration-source registry (`gold_reference`, `stored`, `probe_rotation`);
θ source switch; tilt-nuisance option; `sensitivity.py`; Phase-1 diagnostics.
Gates: synthetic HR-Si and doped Si recover planted values with background ≥ 10% of signal and 30 fs
drift; ±M fit ≡ explicit differencing to 1e-12 on 0/90/180/270; probe-rotation κ = gold κ to < 0.1%;
§8.1 affine map reproduces "42° analysed as 45° → Si 15.97".

**Phase 2 — bench, isotropic (the MVP's step 3, now with the magnet scheme).**
Plan §9 setup steps 1, 2, 5, 6: gold κ, HR-Si, doped Si; MVP gates V6–V8 (doped Si vs transmission on
the same wafer; remount repeatability); §5.3 tilt test as a multi-dataset stage. Output: the decision
on whether the method replaces mirror referencing for isotropic samples.

**Phase 3 — GE core (no beam time).**
`detection.py` D-matrix, `jones.py`, `magnet_sweep_gold` calibration (fits γ, φ₀, β₀ globally and
E_p/E_s per frequency, §7.5), probe-power normalization check, GE simulator, `off_diagonal_floor` and
`probe_walk_signature` diagnostics.
Gates: synthetic gold/Si in GE return off-diagonals at the noise floor and r_pp/r_ss equal to isotropic
mode; planted γ, φ₀, β₀ recovered; a planted HWP wedge is detected by the 90°/360° periodicity test.

**Phase 4 — anisotropic inversion routes.**
Lift Berreman semi-infinite solver from the prototype into `core/anisotropic.py` with its Fresnel
check as a unit test; aligned closed forms (§7.2) as a second unit check of the solver at ψ = 0°, 90°;
routes `in_plane_uniaxial` (per-frequency ε_o, ε_e, global ψ, seeded by the isotropic inversion of
r_pp/r_ss) and `aligned_azimuth_pair`.
Gates: planted uniaxial material recovered at ψ = 30°, 45°; the two routes agree (§7.7); identifiability
reported per frequency (Jacobian conditioning) so weak bins are flagged, not trusted.

**Phase 5 — out-of-plane route.**
`out_of_plane_with_transmission`: load ε_∥ from a transmission result bundle, interpolate onto the
ellipsometry grid, closed-form ε_⊥ (§7.2) with propagated errors (expect 2–3× amplification).
Variable angle of incidence (§7.9) is **deferred** — the 45° fixed geometry has no movable arm; the
route registry makes it a later addition, not a refactor.

**Phase 6 — integration.**
Bundle export + catalog indexing, quantity registration so results open in the results viewer, Drude
post-fit, `docs/` page for lab users (how to name files, which mode, what the diagnostics mean).

Rough size: Phases 0–1 and 3–4 are the substantial code; 2 is bench time; 5–6 are small.

---

## 7. Testing

Per the house rule, every phase has both:
- **Unit tests** per core module, each against a known answer (closed forms, Fresnel limits,
  planted parameters, convention signatures).
- **Headless workflow tests** per mode × route: simulator writes real `.acc` files → the run_me's
  `main(["--simulate", ..., "--no-graph"])` → assertions on the exported bundle and on the
  diagnostics that should (and should not) fire. Includes fault-injection runs (blocked magnet
  state, missing probe setting, degenerate probe angle, large drift) to prove the diagnostics catch
  them.

---

## 8. Open questions (none block Phases 0, 1, 3)

1. Is the GaP already moved off the degenerate azimuth to ~31.7° from [001]? (Prerequisite for Phase 2.)
2. Magnet rotation hardware: discrete 0/90/180/270 stops or continuous? (Continuous enables the §7.5
   sweep; discrete is enough for isotropic mode.) Is the state recorded in the .acc header, or only
   the filename?
3. Probe rotation for GE / probe-rotation calibration: HWP in the 800 nm arm or liquid-crystal
   retarder? (Drives how hard the §7.8 wedge test needs to be.)
4. Doped Si resistivity and whether a same-wafer transmission measurement exists (truth standard).
5. Target anisotropic sample for Phase 4 validation — a known in-plane uniaxial crystal (e.g. a
   quartz or sapphire a-cut plate) before CNT?
6. Rename confirmation: `thz_ellipsometry/` + `thz_ellipsometry_run_me.py` (recommended), or keep
   the existing names.

---

## 9. As built: where Phases 0–1 departed from this plan, and why

Design deviations only (progress lives in `~/.claude/global_projects.md`, not here).

- **Window shape is a flat-top Tukey, not Hann** (`core/preprocess.py`, `preprocess.window_shape`).
  Under a sloped window a drifting pulse also changes amplitude, so the drift is not a pure phase
  ramp for any pulse off the window centre. Hann + strong background gave harmonic reduced
  chi-square 9.1; Tukey 1.11. Lab notebook F38.
- **Noise from `drift_corrected_scatter`, not the three-term `fit_noise_parameters`.** The latter
  under-reports sigma_alpha at 4–8 repeats (0.30–0.65x). F38, OQ13.
- **A known magnet offset is removed in the loader** (`geometry.magnet_angle_for_p_deg`), which is
  algebraically identical to undoing the Moebius mixing; there is no separate `emitter_offset_deg`
  key. The Moebius machinery remains for the two-reference fit of an UNKNOWN offset.
- **The acquisition-mode registry lives in `adapters/stages.py`**, beside the stage registry, rather
  than in its own `acquisition_modes.py`; `conventions.py` was not needed (the convention is stated
  in `core/model.py` and enforced by tests); `export.py` is Phase 6.
- **The tilt fit reports a rank-deficient Jacobian as infinite error.** A pseudo-inverse had
  returned 0.24 +/- 0.001 deg for a true 0.5 deg on a flat sample.
- **Calibration error is not in the per-frequency bars.** C is one number per run, so its error is
  fully correlated across frequency — a run systematic, carried as `ChannelCalibration.standard_error`.
- **Ellipsometry diagnostics share the repo's `@diagnostic` registry** under `ellipsometry.*` stage
  names and appear in `docs/assumptions_ledger.md`. Regenerating the ledger must import
  `thz_ellipsometry.adapters.diagnostics`, or those entries drop out.
- **Magnet angle: a calibration table, not a fitted nuisance** (`core/emitter.py`,
  `geometry.magnet_calibration` = {polarisation: reading}). Per-state angle errors are not
  identifiable from sample data (F39). Replaces the single `magnet_angle_for_p_deg` offset; one
  entry is that offset.
- **Amplitude drift nuisance** (`acquisition.amplitude_model`, default `linear_ramp`), fitted with the
  delay by the same variable projection.
- **Bench tools** (`core/bench.py` pure; `adapters/bench_tools.py` with a `@bench_tool` registry;
  `thz_ellipsometry_bench_run_me.py`): wire-grid nulls (paired sweeps share a background), purge
  settling, HWP walk (plan sec. 7.8). These were not in the plan; they serve the bench sessions.

