# Assumptions ledger

**Generated from `dataset_core.adapters.diagnostics`. Do not edit by hand** —
edit the `@diagnostic` registration and regenerate, so the runtime check and
this document can never disagree.

Each entry is something a pipeline stage takes for granted. Every one of them
is checked against real data at run time; none of them halts a run.

## Stage: `preprocessing`

### the individual scans behind each averaged trace are still available

- **Check:** `per_scan_data_survived` (severity when broken: **FAIL**)
- **Why it matters:** Every repeat-based estimate — noise amplitudes, drift, spectral error bars — reads the per-scan matrix. When a row-count-changing step drops it, the matrix silently becomes the averaged trace as a single 'scan', whose scatter is zero. Nothing errors; the estimates just quietly describe nothing. This is how it was being lost by the very first pipeline step.
- **If it fails:** the step that changed the row count must call thz_adapter.carry_scan_matrix with its own transform; until then, treat any repeat-based number for this file as unavailable rather than as a small value

## Stage: `noise`

### there are enough repeats for the scatter between them to mean anything

- **Check:** `enough_repeats_for_noise` (severity when broken: **warn**)
- **Why it matters:** A variance estimated from M samples carries a relative error of 1/sqrt(2(M-1)) — about 40% at M=4, 13% at M=30. Below roughly eight repeats the noise estimate is itself noise. Note this depends only on the NUMBER of repeats, not on how noisy each one is: a single scan being too weak to interpret alone is not an obstacle, because the DFT is linear.
- **If it fails:** acquire more scans, or quote the estimate with its own uncertainty attached

## Stage: `acquisition`

### the instrument held still over the acquisition

- **Check:** `acquisition_drift` (severity when broken: **warn**)
- **Why it matters:** Amplitude and arrival time drift over a run — on our purge box, 5% and 9 fs over 80 minutes while the nitrogen equilibrates. Drift is not removed by averaging, and a measurement can be drift-limited rather than noise-limited, in which case acquiring more scans does nothing at all. It also inflates any naive repeat-scatter estimate that does not fit it out first.
- **If it fails:** wait for the purge to settle, or interleave sample and reference A-B-A-B so the drift affects both equally; check drift_inflation before trusting more averaging

### the purge has equilibrated before the acquisition started

- **Check:** `purge_still_equilibrating` (severity when broken: **warn**)
- **Why it matters:** Water vapour absorbs across the THz band, and its continuum absorption grows with frequency. So while a nitrogen purge is still displacing room air, the measured spectrum RISES, and rises MORE at high frequency. A measurement taken through that transient has a sample and a reference recorded in different atmospheres, which is a systematic no amount of averaging removes. 

Crucially this does NOT require resolving the water lines: unresolved is not invisible, because a line narrower than the resolution still removes its energy from the band. Measured on a 111-scan CNT acquisition over 83 minutes, the band gain went +2.0% at 0.4-1 THz, +7.0% at 2-3 THz, +12.9% at 3-4 THz and +17.1% at 4-6 THz, while a settled reference acquisition on the same instrument was flat. 

The TILT is what identifies water specifically: a laser power drift scales the whole spectrum uniformly and leaves the tilt unchanged, so a frequency-dependent gain cannot be explained that way.
- **If it fails:** wait for the purge to settle (99% settling has been measured at ~3.7 hours, not the assumed 15 minutes), or interleave sample and reference A-B-A-B so both see the same atmosphere

## Stage: `transfer_function`

### the band the noise floor is read from has reached a noise plateau

- **Check:** `noise_floor_band_is_a_plateau` (severity when broken: **warn**)
- **Why it matters:** The tail-median floor is a DYNAMIC RANGE metric (Naftaly & Dudley 2009), and it only measures noise if the spectrum has flattened out in the band it is read from. On a short record with a sharp pulse it has not — real pulse content persists to the sampling limit — so the 'floor' is set by signal. Measured on our CNT data it sat 8-38x above the true random scatter, did NOT fall as 1/sqrt(M) when more scans were averaged, and reported a session that was 2.3x quieter as 3.9x worse. Because the transfer error is built as floor/|Y|, it then grows toward high frequency for reasons unrelated to the measurement.
- **If it fails:** estimate the uncertainty from the repeat scans instead (thz_core.noise); treat any floor-derived error bar as an upper bound in the meantime

## Stage: `window`

### the analysis window fits inside the recorded trace

- **Check:** `window_fits_the_record` (severity when broken: **warn**)
- **Why it matters:** A window wider than the record is clipped at the trace edge, so it does not return to zero and the effective apodisation is not the one requested. The record length also sets the true frequency resolution (1/T), so a clipped window means the resolution being quoted is not the resolution being measured.
- **If it fails:** reduce half_width_ps, or acquire a longer trace

## Stage: `resolution`

### plotted and stored spectra are on the resolution actually measured

- **Check:** `spectra_are_oversampled` (severity when broken: **info**)
- **Why it matters:** Zero-padding to a large n_fft makes the BIN SPACING far finer than the RESOLUTION, which is fixed at 1/T by the windowed record length. The extra points are sinc interpolation, not measurement, and reading structure from them reads structure from the padding.
- **If it fails:** markers are placed on the independent grid automatically; set config['resolution']['limit_to_instrument_resolution'] to decimate the stored arrays too
