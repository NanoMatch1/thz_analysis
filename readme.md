# matchbook/thz — THz-TDS Reflection Analysis Framework

A batch analysis pipeline for THz time-domain spectroscopy (TDS) with a focus
on the **window-coupled reflection geometry** used at IMDEA Nanociencia. Built
around the `thz-core` scientific library (sibling directory `matchbook/thz-core`),
this repo adds dataset orchestration, preprocessing, file I/O, visualisation, and
interactive tools needed to go from raw `.acc` multi-scan acquisitions to n, k,
ε, and σ(ω).

**Companion documents** (read these alongside this file):
- `docs/cnt_reflection_measurement_brief.md` — **why the CNT reflective-sample measurement
  is hard**, every geometry/trick tried, and where it stands. Start here for the CNT
  problem; `docs/README.md` maps the full documentation set.
- `ANALYSIS_NOTES.md` — physics decisions, gotchas, validation results, and the
  rationale behind every non-obvious design choice. Intended to seed future
  tutorial / user-guide docs. **Read this before touching inversion or phase handling.**
- `TODO.md` — tracked outstanding work with full context.

---

## Goal

Extract complex optical constants n(ω) − ik(ω) and optical conductivity σ(ω) of
conductive thin films (primarily CNT papers) from THz reflection measurements
taken in the window-coupled geometry: sample pressed against the flat back face of
a SiO₂ window; THz beam enters at 45° external incidence (s-pol), refracts to
≈21.3° internal (see `ANALYSIS_NOTES.md §1`), and the SiO₂→sample interface
carries the sample information. A SiO₂-only reference cancels the window path.

The secondary goal is a validated, self-documenting framework that supports
reproducible experimental measurements and can be extended to new geometries.

---

## Repository structure

```
matchbook/thz/
├── run_me_dataset_core.py      — main pipeline script (edit file_dir + run)
├── ANALYSIS_NOTES.md           — physics decisions and methodology log  ← READ THIS
├── TODO.md                     — tracked open work
├── open_session.py              — reopen a saved .thzbundle via the catalogue; interactive
│                                  REPL / CLI to browse, replay, fit, export, extract quantities
├── dataset_core/
│   ├── adapters/
│   │   ├── thz_adapter.py      — THE main file: bridges DataSet ↔ thz-core
│   │   ├── display.py          — reusable registry-driven plotting: plot_quantity,
│   │   │                          plot_time_domain, get_series (see docs/quantity_registry.md)
│   │   ├── quantity_registry.py — single source of truth for displayable/exportable quantities
│   │   ├── session_bundle.py / catalog/ — .thzbundle save/load + rebuildable catalogue index
│   │   └── fitting.py / conductivity_fitting.py — Drude/Drude-Smith fitting, MC error bars
│   ├── data_structures/
│   │   └── thz.py              — THzData container (multi-scan + averaged working trace)
│   ├── dataset.py              — DataSet, DataService, file loading, grouping
│   ├── io/loaders/             — .acc, .dat, .txt, .csv loaders
│   └── services/grouping.py   — filename-token-based pairing (sample→reference)
└── tests/
    ├── test_grouping_tiebreaker.py          — 9 tests (filename grouping)
    ├── test_reflection_pipeline_workflow.py — 5 tests (end-to-end + sub-sample)
    ├── test_segment_preserves_scans.py      — 2 tests (scan preservation)
    ├── test_time_shift_sweep.py             — 6 tests (time-shift sweep/slider)
    ├── test_window_selfref_workflow.py      — 5 tests (self-referencing + characterise_window)
    └── test_display.py                      — 17 tests (item selection, style resolution,
                                                 data access, headless figure-build workflow)

matchbook/thz/thz_core/        — nested thz-core repo clone (gitignored here,
                                  tracked as its own repo); imported as
                                  `thz_core.thz_core`
    thz_core/
    ├── preprocess.py           — baseline, alignment, windowing, padding
    ├── fft.py                  — rfft with explicit sign convention
    ├── transfer.py             — H = Y_samp/Y_ref, trusted_band_mask,
    │                             remove_phase_offset (phaseex), phase_ramp
    ├── invert.py               — invert_nk (transmission), invert_nk_reflection
    ├── unwrap.py               — robust_unwrap (quality-guided, SNR-weighted)
    ├── derive.py               — eps, sigma from n, k
    ├── multilayer.py           — Fresnel coefficients, Snell's law
    ├── kramers_kronig.py       — KK phase correction (prototype; see ANALYSIS_NOTES §8)
    ├── window.py               — window model + n_SiO₂ inversion from W=Y₂/Y₁ (§11)
    └── fit_gui.py              — Drude / Drude–Smith fitting GUI
    tests/                      — 175 passing tests
```

---

## The two-phase pipeline

### Phase 1 — Preprocessing and segmentation (run once per measurement set)

Run on the parent directory containing raw `.acc` files. The `preprocess()`
function in `run_me_dataset_core.py` wraps this:

```python
fileDir = r"C:\...\CNT-13\A"
dataset = DataSet(fileDir)
dataset.load_all_data(case_insensitive=True)
dataset.group_files(keywords=['type'])          # pairs sample_* → reference_*
thz.subtract_baseline(dataset)
thz.align_to_reference(                         # see ANALYSIS_NOTES §4
    dataset, ref_type="reference",
    roi=(152, 156),                             # ps, restrict to front-pulse region
    subsample_correction=True)                  # sub-sample phase ramp in transfer_function
thz.normalise(dataset, config={"bounds": (152, 156)})
thz.segment_reflections(dataset, segments={
    'first_reflection':  (151, 158.8),          # air→SiO₂ front face pulse
    'second_reflection': (162.2, 168.2),         # SiO₂→sample back face pulse
})                                              # see ANALYSIS_NOTES §2
sys.exit()   # re-run pointing fileDir at segmented/second_reflection/
```

`segment_reflections` is a **clean splitter**: it crops each file to the time
gate and writes every individual acquisition (not just the mean) into
`segmented/<component>/<original_filename>.acc`. Original filenames are preserved
so grouping on the reloaded files is identical.

### Phase 2 — Spectral analysis (run on the segmented directory)

```python
fileDir = r"C:\...\CNT-13\A\segmented\second_reflection"
dataset = DataSet(fileDir)
dataset.load_all_data(case_insensitive=True)
thz.subtract_baseline(dataset)
dataset.group_files(keywords=['type'])
thz.global_truncate(dataset)                    # common time range across all files
thz.window_time(dataset, config={"window": {"type": "hann"}})
thz.zero_pad(dataset, config={"pad": {"extend_factor": 3.0}})
thz.fft_spectrum(dataset)
# SNR mask config MUST travel with transfer_function — see ANALYSIS_NOTES §5 / SNR note below
# self_reference: front-pulse drift correction from the sibling
# first_reflection folder — see ANALYSIS_NOTES §11
thz.transfer_function(dataset,
    config={"transfer": {"apply_snr_mask": True, "self_reference": True},
            "mask": {"snr_thresh_db": 2, "tail_fraction": 0.25, "min_contiguous_bins": 3}},
    ref_type='reference')
thz.invert_nk_reflection(dataset,               # see ANALYSIS_NOTES §1 + §6b
    geometry="window", theta_deg=45,
    polarization='s', n_window=1.95)
thz.derive_eps_sigma(dataset)                   # see ANALYSIS_NOTES §3 for eps_background
thz.result_viewer(dataset)
```

### Transmission flow (free-standing slab) — `ANALYSIS_NOTES §18`

No segmentation. A single pulse per trace; the sample pulse is delayed relative to
the air/open-beam reference by the material group delay `(n−1)d/c`.

```python
dataset = DataSet(fileDir)
dataset.load_all_data(case_insensitive=True, explicit_dir=True)
dataset.group_files(keywords=['type'])          # pairs sample → reference
thz.subtract_baseline(dataset)
thz.centering_manual(dataset)                   # window hygiene: peaks to a common
                                                # array index, equal start→peak→end.
                                                # Each file now starts at a different
                                                # absolute t[0] — that difference IS
                                                # the group delay (§18).
thz.window_time(dataset, config={"window": {"type": "tukey", "alpha": 1}})
thz.zero_pad(dataset, config={"pad": {"extend_factor": 1.0}})
                                                # zero_pad runs align_to_common_time_axis
                                                # first → lays traces on ONE shared axis,
                                                # converting the t[0] differences into the
                                                # exp(-iωΔt) phase ramp the FFT needs.
thz.fft_spectrum(dataset)
thz.transfer_function(dataset,                  # transmission: self_reference=False
    config={"transfer": {"apply_snr_mask": True, "self_reference": False},
            "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3}},
    ref_type='reference')
thz.remove_phase_offset(dataset,                # phaseex: subtract H's phase intercept
    config={"phase_offset": {"band_thz": (0.3, 2.0)}})   # sub-2π residual only (§18)
thz.invert_nk(dataset, thickness_m=2.08e-3)     # anchor_phase_origin=True (default) removes
                                                # the whole-cycle (2π) wrap that droops n at
                                                # low f for thick samples — the main droop fix
```

Group-delay preservation is **structural**, not an explicit phase factor: `centering_manual`
leaves each file with a different absolute `t[0]`, and `align_to_common_time_axis`
(inside `zero_pad`) re-lays them on a shared axis so the offset becomes the FFT phase
ramp — the equivalent of legacy `phioffset` (`ANALYSIS_NOTES §18`). The low-frequency `n`
droop is a **whole-cycle (2π) wrap error** for thick samples (the phase exceeds π before
the first reliable bin); it is fixed by `invert_nk`'s `anchor_phase_origin` (default True),
because an integer cycle cannot be carried through the complex `H` that `remove_phase_offset`
edits. `remove_phase_offset` then removes the remaining sub-2π residual.

---

## Key design decisions

### Geometry: window-coupled reflection (`ANALYSIS_NOTES §1`)
External 45° → internal ≈21.3° (Snell, n_SiO₂≈1.95). Fresnel inversion uses
`n_incident = n_SiO₂` at the internal angle — using air at 45° is wrong for this
geometry. `r_sample = r_{SiO₂→air}·H`; the computed Fresnel coefficient restores
absolute scale. The SiO₂ window improves inversion conditioning for conductive
samples: using n_inc=1.95 keeps `|1+r|` bounded where air/gold geometry would
blow up (r→−1 problem, `ANALYSIS_NOTES §6b`).

### T0 alignment — sub-sample split (`ANALYSIS_NOTES §4`)
`align_to_reference` splits the cross-correlation shift into a whole-sample part
(slides the time axis, exact, no resampling) and a sub-sample residual (|Δ|≤dt/2)
applied in `transfer_function` as an exact spectral phase ramp `H·exp(−iω·Δt)`.
This avoids time-domain interpolation (lossy low-pass) and silent quantisation by
`pad_to_common_grid`. Toggle: `SUBSAMPLE_TIMING_CORRECTION` constant in
`thz_adapter.py`.

### Per-scan statistical power through segmentation
`THzData.raw_data` = `[time_ps, scan1..scanN]`. A parallel working matrix
`processing_dict['working_scans']` is kept in lockstep through the three linear
preprocessing steps (subtract_baseline, align, normalise). `segment_reflections`
crops this matrix and writes every scan. The averaged result is bit-identical
(Δ≤1.1×10⁻¹⁶). Previously, segmentation cropped the averaged `data` array and
the output `.acc` files held `[mean, stderr]` as two fake scans — destroying
statistical power.

### SNR mask
`transfer_function` builds the trusted-band mask from the intersection of sample
and reference spectral dynamic range. **The mask thresholds must be passed to
`transfer_function`** — a standalone `trusted_band_mask` call before it is a
no-op (H not yet computed). The mask drives inversion and dims plots. Running at
the core default (20 dB) vs 2 dB changes the trusted bandwidth by roughly 2×
(50 vs 104 bins on CNT-13A data). See `ANALYSIS_NOTES §5`.

### Front-pulse self-referencing (`ANALYSIS_NOTES §11`)
The first reflection (air→SiO₂ front face) never sees the sample, so the
intra-trace ratio `W = Y₂/Y₁` of a bare-window trace is a window-only transfer
function (source/detector/air-path cancel). With
`transfer_function(..., config={"transfer": {"self_reference": True}})` the
pipeline computes `H = (Y₂ₛ/Y₁ₛ)/(Y₂ᵣ/Y₁ᵣ)` using the sibling
`first_reflection/` segment folder — each trace referenced to its own front
pulse, immune to mount-to-mount drift (window deformation under pressure,
rotation, realignment). Validated on CNT-13/D: 7–10% rms structured drift
replaced by a ~1.6% noise floor; repeat spread of σ₁ drops 18%→5%.
`thz.characterise_window(first_path, second_path, thickness_m=0.9e-3,
theta_deg=45)` extracts n_SiO₂(ω) from the same W (measured: 1.962–1.967 flat).

### Phase unwrap (`ANALYSIS_NOTES §7b`)
`thz_core.robust_unwrap` excludes low-SNR bins from the ±2π decisions. Key
insight: `np.unwrap` is direction-independent and step-local — starting at a
high-SNR bin helps nothing; only *excluding* noisy bins from the decision chain
works. With no weights it is byte-for-byte `np.unwrap`.

---

## Geometries supported

| Geometry | `geometry=` | Reference | Incident medium | Use case |
|----------|-------------|-----------|-----------------|----------|
| Window-coupled | `'window'` | Computed r_{SiO₂→air} | SiO₂ (n≈1.95) | CNT paper, films on glass |
| Gold mirror | `'gold'` | r_gold ≈ −1 | Air | Flat solid reflectors (Si) |
| Transmission | `invert_nk` (separate path) | Open beam or substrate | Air | Thin slabs, substrates |

---

## Validation status (`ANALYSIS_NOTES §6, §6b`)

| Test | Result | Notes |
|------|--------|-------|
| Si (gold/air, 45°, s-pol) | n≈3.35 ✓ | Near literature 3.418; σ noisy — reflection is a weak σ-probe for low-loss Si |
| Doped Si transmission (Chris) | n≈3.02, clean Drude σ₁/σ₂ crossing at 0.96 THz ✓ | Requires `eps_background=11.7` (`ANALYSIS_NOTES §3`) |
| CNT window reflection | n≈4.5 flat, stable σ ✓ | Air/gold CNT blows up; window geometry essential |
| Sub-sample ramp A/B | ON beats OFF ✓ | `test_reflection_pipeline_workflow.py` |
| Segment scan preservation | Δmean=1.1×10⁻¹⁶, full scan count preserved ✓ | `test_segment_preserves_scans.py` |
| Front-pulse self-referencing | repeat spread n 11%→4%, σ₁ 18%→5% ✓ | `test_window_selfref_workflow.py` + CNT-13/D (`ANALYSIS_NOTES §11`) |
| n_SiO₂ from single bare-window trace | 1.962–1.967 flat, 0.5–2.7 THz ✓ | `characterise_window`, thz-core `test_window.py` |

**Test counts:** dataset_core 35 pass (reflection workflow/self-ref/time-shift/segment/
container suites), thz-core 180 pass (incl. `test_remove_phase_offset.py`).

---

## Known issues

### Active
- **Sub-sample residual lost across the two-phase boundary** — `align_to_reference`
  (phase 1) stores the residual in `processing_dict`, which does not persist
  through `segment_reflections`; phase-2 `transfer_function` therefore never
  applies the §4 phase ramp in the split workflow. Self-referencing mode
  (`ANALYSIS_NOTES §11`) carries the front-pulse timing itself and is unaffected;
  the conventional path needs the residual persisted (e.g. a sidecar file or
  header token) or re-measured in phase 2.
- **Window gating spans the full segment** — `core.window_time` has no pulse-centred
  width-based gate. The `alpha` kwarg is silently ignored with `type="hann"` (Hann
  takes no taper parameter). Currently the Hann is applied to the whole segmented
  trace. See `TODO.md` for the planned `gate_width_ps` fix.
- **Contact gap (CNT-on-glass) — the dominant limit.** A ~1–20 µm air gap between the
  pressed CNT paper and the window's back face turns the measured reflection into a
  Fabry–Pérot etalon rather than a single SiO₂→CNT interface — confirmed instrumental via
  a flat linear phase in φ(H), not a code bug (`ANALYSIS_NOTES §9/§9b`); this is what
  drives CNT reflection **n below 1 at high frequency**. Fabry–Pérot de-embedding is
  implemented (exact single-gap math + an interactive position/width slider explorer wired
  into `run_me_low-level.py`) and validated on **silicon controls in both s- and
  p-polarisation** (recovers n≈3.418; gaps measured per-mount at 1–3 µm). **CNT-21 verdict
  (2026-07-02):** the instrument itself is good to 0.5–2%, but every de-embed model —
  single-gap, statistical-gap FP, graded EMT — stalls at a smooth ~10% mount-coupling
  systematic that isn't a gap-model problem; the fibre-anisotropy signal (σ∥ ≫ σ⊥, 2–7×) is
  robust regardless. The path past this ceiling is sample presentation (HR-Si deposited
  route), not more inversion math — see `reports/CNT_measurement_lab_notebook.md` (F1–F29)
  for the full trail.

### Open work (`TODO.md`)
- Peak-centred time gating (`gate_width_ps`).
- Wire SNR weights into `invert_nk` so `robust_unwrap` is on by default.
- Fabry–Pérot de-embedding for air-gap removal (synthetic prototype done; real-data + pipeline pending).
- Single-step phase-offset handling — collapse the complex-H round-trip so the integer
  cycle is not lost in `invert_nk`'s re-unwrap; keep BOTH the `anchor_phase_origin` and
  `remove_phase_offset` corrections (parsimony refactor, not a correctness fix).
- `save_database`/`load_database` cleanup (hardcoded path, deduplication).

### Done (was deferred)
- ~~n_SiO₂ extraction from the SiO₂-only trace~~ — `characterise_window` /
  `thz_core.invert_window_index` (`ANALYSIS_NOTES §11`).
- ~~Front-face self-referencing~~ — `transfer_function` `self_reference` config
  option (`ANALYSIS_NOTES §11`).
- ~~Low-frequency `n` droop / residual constant phase offset on H~~ —
  `thz_core.remove_phase_offset` + adapter `thz.remove_phase_offset` (the legacy
  `phaseex` intercept removal, done deterministically); group delay preserved
  structurally by `align_to_common_time_axis` (split out of `zero_pad`).
  See `ANALYSIS_NOTES §18`.
- ~~p-polarisation inversion~~ — closed-form p-pol root-picker in `thz_core.reflection_gap`
  + `thz.fit_dual_pol_reflection`; validated on silicon in both polarisations as part of the
  air-gap de-embed work above (F21/F22).

### Deferred
- Geometry registry (replace `'gold'`/`'window'` string dispatch).
- Per-frequency uncertainty propagation.
- Feed measured n_SiO₂(ω) (from `characterise_window`) into
  `invert_nk_reflection(n_window=...)` as a per-frequency array.

---

## For AI agents

### Files to read first
1. `docs/cnt_reflection_measurement_brief.md` — **why measuring highly reflective (CNT)
   samples is hard**, every geometry/trick tried, and the current verdict. The single
   orientation doc for the CNT reflection problem; read this to get up to speed before
   evaluating a technical idea. (`docs/README.md` maps all documentation.)
2. `ANALYSIS_NOTES.md` — **required** before modifying inversion, phase handling,
   or geometry logic
3. `dataset_core/adapters/thz_adapter.py` — every pipeline step lives here
4. `run_me_dataset_core.py` — current working pipeline

### Architecture rules
- `thz_core` is pure science (arrays in, arrays out, no file I/O, no DataSet).
- `thz_adapter.py` is the only bridge: extracts arrays from `data_obj.data` /
  `processing_dict`, calls `thz_core`, writes results back.
- `data_obj.data` is **always** the averaged `[time_s, mean, (stderr)]` array.
  `data_obj.raw_data` is `[time_ps, scan1..scanN]`. Don't conflate them.
- `processing_dict['working_scans']` is the per-scan working matrix; only consumed
  by `segment_reflections` — don't touch it elsewhere.
- The SNR mask (`transfer_mask`) must be built **inside** `transfer_function`.
  Calling `trusted_band_mask` before it is a no-op.
- Unit convention: `thz_core` operates in SI (seconds, Hz). The adapter converts
  to/from display units (ps, THz) only at plot and input boundaries. Source `.acc`
  files store time in picoseconds; `THzData._average_data()` multiplies by 1e−12 on
  load.
- `SUBSAMPLE_TIMING_CORRECTION` in `thz_adapter.py` is a deliberate single-line
  toggle. Do not silently collapse it to always-on.

### Running tests
```bash
# dataset_core (from matchbook/thz/):
.venv/Scripts/python.exe tests/test_grouping_tiebreaker.py
.venv/Scripts/python.exe tests/test_reflection_pipeline_workflow.py
.venv/Scripts/python.exe tests/test_segment_preserves_scans.py
.venv/Scripts/python.exe tests/test_time_shift_sweep.py

# thz-core (from matchbook/thz-core/):
.venv/Scripts/python.exe -m pytest -q
```
