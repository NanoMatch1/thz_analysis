# TODO

## dataset_core

- [ ] Clean up `save_database` / `load_database` in `dataset.py`:
  - Remove hardcoded `database_dir` default path (`C:\Users\Samuel\Data\database`) — should be injected or read from config.
  - Deduplicate pickle serialisation/deserialisation logic shared with `save_state` / `load_state`.

## thz-core

- [ ] **SNR-anchored phase unwrap** in `invert.py` (and reuse in `kramers_kronig.py`).
  Currently the unwrap anchors at the *lowest finite positive-frequency bin*
  (`invert.py:278-284`). That bin can sit in a low-SNR region where the phase is
  noisy, so a single corrupted sample early in the span propagates a 2π error
  through the rest of the unwrap. Instead: anchor the unwrap at the **highest-SNR
  bin** (peak |H| or peak reference spectrum, within the trusted band) and unwrap
  **outward in both directions** from there — toward DC and toward high f — so the
  branch is set by the most reliable phase and never seeded from noise.
  - Keep the low-frequency absolute-branch anchoring goal (don't lose the wrap
    count for thick samples); the high-SNR start fixes *which* bin seeds the
    branch, the existing logic fixes the *absolute* offset — reconcile the two.
  - Make it a shared helper (e.g. `unwrap_from_anchor(phase, weights, anchor_idx)`)
    so both `invert_nk` and the KK estimator use the same robust unwrap.
  - Add a test: inject noise into low-frequency bins and confirm n is unchanged
    vs the clean case (the current low-anchor version would fail this).
