# TODO

## dataset_core

- [ ] Clean up `save_database` / `load_database` in `dataset.py`:
  - Remove hardcoded `database_dir` default path (`C:\Users\Samuel\Data\database`) — should be injected or read from config.
  - Deduplicate pickle serialisation/deserialisation logic shared with `save_state` / `load_state`.

## thz-core

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
