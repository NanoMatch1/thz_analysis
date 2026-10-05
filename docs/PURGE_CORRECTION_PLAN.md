# Purge-drift correction for any THz-TDS series — plan

*2026-10-05. Queued, not started. Builds on lab notebook F31, F32, F40 and
`explorations/thz_ellipsometry/purge_spectral_template.py`. Status lives in
`~/.claude/global_projects.md`; the daily queue in `~/.claude/WORKING.md`.*

## Goal

A post-processing step that removes the nitrogen-purge transient from ANY measurement series
taken on the instrument with **one box closure** (no opening during the series): transmission,
reflection, ellipsometry, other users' experiments. Output is corrected data plus a full
provenance record, so the correction is reproducible and visible all the way to the catalogue.

## What is known (F32, F40)

- After the box is closed the purge changes the beam path in three ways, all with one spectral
  shape: a **delay** (air -> N2 gas exchange), a flat **gain** and a **log-amplitude tilt linear
  in frequency** (water; the lines are unresolvable on our short records). Together they describe
  the change to 0.3% rms; one SVD mode carries 98%.
- Each settles exponentially. Time constants differ with the disturbance: ~39-44 min after a full
  opening, 8-13 min in another run (fast gas exchange vs slow wall-water desorption) — so they are
  FITTED, never assumed.
- Correcting is better than waiting: with the exponential models, a series starting 10 min after
  closing is recovered to the noise.

## How it works

1. **Per scan**, against a reference scan of the same file: delay (cross-spectrum phase slope),
   gain and tilt (log-amplitude fit across the band). Reuses `acquisition_tracking` /
   `thz_ellipsometry.core.bench` estimators.
2. **One continuous model for the whole closure**: delay(t), gain(t), tilt(t) each
   `A_x exp(-t/tau_x) + c_file`, with time constants and amplitudes SHARED across every file and
   one free constant per file (that file's own signal). Each file's internal trend constrains the
   same curve, so the drift BETWEEN files is inferred from it — and short files borrow tau from
   long ones (fixes F32's "a short run cannot fit its own tau").
3. **Correct every scan to one common reference time inside the measured span** (interpolation
   only — no extrapolation to a never-reached plateau), so sample and reference end at the same
   purge state. Delay as a sub-sample spectral phase ramp; gain and tilt as a spectral filter,
   **clamped outside the trusted band** so dead high-frequency bins are not amplified.
4. Average the corrected scans; write corrected files.

## Guards (what it cannot do, and how it says so)

| Risk | Guard |
|---|---|
| A box opening mid-series breaks the curve | Detect timestamp gaps and steps in per-scan delay/gain; refuse to bridge them, or split into segments and report it |
| A sample that genuinely changes looks like purge | Report each file's residual against the purge signature (delay + gain + linear tilt); option to fit the purge on REFERENCE files only (gold / empty path) and apply it to samples — the safer default when sample stability is unknown |
| Too short a series to pin the curve | Report parameter uncertainties and propagate them into the corrected spectra; warn when tau is not constrained |
| Correction amplifying noise | Clamp the filter outside the band; record the clamp |

Never halts silently: every outcome is a recorded verdict.

## Output and provenance

- Corrected `.acc` files beside the originals, new suffix (`<name>_purgecorr.acc`), originals
  untouched; the header points to the sidecar. Every existing run_me consumes them unchanged.
- Sidecar `<name>.purgecorr.json`: input files with SHA-256, scan counts and time spans; model
  names (from the registry) with fitted parameters, errors and taus; the reference time; per-file
  residuals and guard verdicts; the clamp band; full config; git SHAs of thz_analysis and
  thz-core; date. Schema-versioned.
- Bundles record the correction as an input step, so the catalogue shows it.

## Validation

1. Known-answer: simulated series with planted transients, gaps and a box opening, through the real
   `.acc` writer and loader (same pattern as `thz_ellipsometry`).
2. Self-consistency on the August data: split the long 12mm run; early and late halves must agree
   after correction (they differ by up to 20% at 2 THz before).
3. **Decisive bench test (~90 min):** open and close the box, then alternate sample and reference
   with the lateral translation stage — no opening — for ~90 min. The corrected sample/reference
   ratio from the first pairs must equal that from the last; the uncorrected one drifts.

## Structure

- Pure model (fit, apply, guards): one shared module, also used by the ellipsometry harmonic fit so
  the two cannot diverge — destined for thz-core (decision open: commit there directly, or develop
  here under a `core/` boundary and move it, as with `thz_ellipsometry`).
- File handling, sidecar writer and `purge_correction_run_me.py` (config dict) in thz_analysis.
- Unit tests per function + headless workflow tests through real files.

## Open decisions

1. Home of the pure model: thz-core now, or here first?
2. Output: corrected `.acc` + sidecar (recommended) vs an in-memory pipeline stage.
3. Add the 90-min translate-and-alternate validation run to a bench day.
