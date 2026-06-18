# TODO

## dataset_core — two-phase pipeline

- [ ] **Persist the sub-sample alignment residual across segmentation.**
  `align_to_reference` (phase 1) stores `subsample_shift_seconds` in
  `processing_dict`, which is lost when the segmented .acc files are reloaded in
  phase 2 — so the §4 spectral phase ramp has not been applied in the split
  workflow. Options: write the residual into the segmented file header (a
  `%param` token the loader picks up), or a sidecar JSON per segment folder.
  Not needed when `transfer.self_reference` is on (the front-pulse correction
  carries the true timing itself — ANALYSIS_NOTES §11), but the conventional
  path should be correct too.

- [x] **Front-pulse self-referencing + window characterisation** (2026-06-11,
  ANALYSIS_NOTES §11): `transfer_function` `self_reference` config option,
  `characterise_window`, thz-core `window.py` (`window_transfer_model`,
  `invert_window_index`, `envelope_peak_time`). Tests:
  `test_window_selfref_workflow.py` (5), thz-core `test_window.py` (9).
  - Follow-up: feed measured n_SiO₂(ω) into `invert_nk_reflection` as a
    per-frequency `n_window` array.
  - Follow-up: average W over several bare-window acquisitions to push the
    ~1.6% prediction noise floor down.

## dataset_core

- [ ] Clean up `save_database` / `load_database` in `dataset.py`:
  - Remove hardcoded `database_dir` default path (`C:\Users\Samuel\Data\database`) — should be injected or read from config.
  - Deduplicate pickle serialisation/deserialisation logic shared with `save_state` / `load_state`.

## dataset_core / thz-core — windowing

- [ ] **Peak-centered time gating in `window_time`.** `core.window_time` only gates
  via explicit `gate_start` / `gate_end`; there is no width-based gate, so a config
  like `{"window": {"length": 0.3}}` (or `alpha` with `type:"hann"`) is silently
  ignored and the window spans the whole trace. Add a gate specified as a width
  centered on the main pulse: locate the |peak| (reuse
  `preprocess.find_extremum_in_index_range`), then set
  `gate_start = t_peak - width/2`, `gate_end = t_peak + width/2`. Decide the knob:
  `gate_width_ps` (absolute) is clearest; thread it through the adapter
  `window_time` config and add a unit test (gate centered on peak, correct width,
  energy outside the gate zeroed). Keeps the existing explicit-bounds path intact.

## thz-core

- [ ] **Single-step phase-offset handling (collapse the complex round-trip).**
  (ANALYSIS_NOTES §18; `explorations/explore_phase_unwrap_vs_legacy.py`.) Today the
  transfer phase makes a complex round-trip: `transfer_function` unwraps and stores
  H **complex**, then `invert_nk` re-unwraps `np.angle(H)`. That re-unwrap discards
  the integer-cycle (2π) part of any phase-offset correction, so we need **two**
  mechanisms where legacy needs one: the `invert_nk` `anchor_phase_origin` (integer
  cycle) **plus** `remove_phase_offset` (sub-2π residual). The legacy `phaseex`
  removes the full intercept once, on the **real** unwrapped phase, immediately
  before `n = 1 − cφ/(ωd)`, never re-wrapping. Simplification: remove the intercept
  once on the real unwrapped phase and carry that phase to `invert_nk` without
  re-unwrapping (e.g. pass an already-unwrapped phase, or have `invert_nk` accept a
  precomputed phase and skip its own unwrap). Verified equivalence first: the anchor
  alone already flattens n (~1.94, matching legacy+phaseex); this is a parsimony /
  maintainability change, not a correctness fix.
  - **KEEP BOTH the `anchor_phase_origin` integer-cycle fix AND `remove_phase_offset`**
    (Samuel, 2026-06-18): the sub-2π residual correction is wanted even though it is
    small. The goal of this item is to remove the *re-unwrap that loses the integer
    cycle*, not to drop either correction. After the change both should still apply —
    just on a phase representation that no longer round-trips through complex H.
  - Keep `anchor_phase_origin` working for callers that still pass complex H. Touches
    `fft_spectrum` / `transfer_function` / `invert_nk` phase flow — gate it and re-run
    both suites + the exploration.

- [x] **SNR-guided phase unwrap** — `thz_core/unwrap.py::robust_unwrap` (2026-06-09).
  Shared helper used by `invert_nk` (new `snr_weights=` kwarg) and the KK estimator
  (`estimate_misplacement` / `correct_reflection_phase`, new `snr_weights=`).
  - **Key correction to the original idea:** merely *starting* `np.unwrap` from a
    high-SNR bin does **not** help — each ±2π decision is step-local and
    direction-independent, so the same steps are crossed and decided identically
    regardless of where you start. The real failure is a *run* of low-SNR bins
    whose noisy phase random-walks across ±π; `np.unwrap` integrates that into a
    net spurious wrap that offsets everything after it.
  - **What was built:** a quality-guided unwrap that *excludes* low-SNR bins from
    the wrap decisions entirely (unwrap across trusted bins only, then snap the
    rest onto that branch). Absolute branch anchored at the lowest *trusted* bin.
  - With `weights=None` it is byte-for-byte `np.unwrap` (no regression). Robustness
    is opt-in via weights; `snr_floor` fraction drops a low-amplitude tail.
  - **Caveat for thick samples:** excluding the low-f bins loses the absolute wrap
    count, so weights must keep enough low-f bins for thick/dispersive transmission.
    The exclusion is safe for reflection/thin samples (phase < π, no real wraps).
  - Tests: `test_robust_unwrap.py` (8), `test_invert_unwrap.py::test_low_frequency_noise_excluded_by_snr_weights`.
  - **Follow-up:** wire a real SNR proxy (reference spectral magnitude / dynamic
    range from `trusted_band_mask`) through the adapter into `invert_nk(snr_weights=)`
    and the KK calls, so the robustness is on by default in the pipeline rather than
    opt-in at the core API.
