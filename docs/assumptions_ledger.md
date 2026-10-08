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

## Stage: `ellipsometry.acquisition`

### the probe polarisation is not at a degenerate azimuth of the EO crystal

- **Check:** `probe_degeneracy` (severity when broken: **FAIL**)
- **Why it matters:** At 0 or 45 deg from [001] one THz component produces no signal at all, so P/Q is a ratio with a zero in it and rho cannot be measured, however long you average.
- **If it fails:** Rotate the probe (or the crystal, once) to about 31.7 deg from [001], where both channels are equally sensitive.

### every acquisition is a pure first harmonic in polarisation angle plus a constant background, up to the fitted drift

- **Check:** `harmonic_model_holds` (severity when broken: **warn**)
- **Why it matters:** Anything outside that form -- a magnet that does not saturate, a drift the ramp cannot describe, a misread filename -- is not cancelled by the ratio and goes straight into rho.
- **If it fails:** Inspect the per-frequency harmonic residual in the run figure; check magnet saturation and the acquisition order; try drift_model='per_acquisition' if the drift is not monotonic.

### the non-magnetic background is small compared with the magnetic signal

- **Check:** `background_is_small` (severity when broken: **warn**)
- **Why it matters:** The background term removes it exactly only if it is truly independent of the magnet state. A large background makes any magnet-dependent part of it (stray field reaching the sample, a substrate signal that depends on M) a first-order error.
- **If it fails:** Find the source: optical rectification in the emitter substrate, pump leakage onto the detector, or electronic pickup. Block the pump to separate the last two.

### the fitted timing drift is well inside the range the drift model searches

- **Check:** `drift_within_range` (severity when broken: **warn**)
- **Why it matters:** A drift that reaches the search bound has been clipped, and the part that was not fitted remains as a phase error in rho (1.5 fs is ~0.9% at 1 THz).
- **If it fails:** Shorten the acquisition or cycle the states faster; check the lab temperature and the purge; raise maximum_drift_fs only if the drift is real.

### the acquisition order lets the drift be told apart from the channel ratio

- **Check:** `drift_separable_from_ratio` (severity when broken: **warn**)
- **Why it matters:** A drift is measured by seeing the SAME polarisation state at different times. In a single pass 0/90/180/270 on gold with C ~ -1 the p and s channels are equal, so 0 and 90 (and 180 and 270) look alike and a drift between them is, to first order, a change of P/Q -- exactly the calibration. The fit then returns a number, but a biased one (F40: index error 0.026 -> 0.14 with no purge at all).
- **If it fails:** Record the block as a palindrome (0,90,180,270,270,180,90,0): it revisits every state and separates drift from C at any channel ratio.

### the fitted amplitude drift across a block is a few percent at most

- **Check:** `amplitude_drift_small` (severity when broken: **warn**)
- **Why it matters:** The amplitude ramp removes a smooth scale change between magnet states. A large one means the laser or the purge changed a lot inside the block, and anything not smooth (a step when the box was opened) is not removed and leaks into rho.
- **If it fails:** Wait longer after closing the box (bench_run_me live), and check the laser power log; repeat the block if it coincides with a disturbance.

### each acquisition has enough repeat scans for the noise model

- **Check:** `noise_model_available` (severity when broken: **info**)
- **Why it matters:** Without the noise model the fit is unweighted and its error bars come from the residual scatter, which is fine for the values but makes the bars themselves rough and loses the chi-square test of the harmonic model.
- **If it fails:** Record at least three scans per acquisition (config['noise']['minimum_scans']).

### the fixed-width window fits inside the recorded trace

- **Check:** `window_inside_record` (severity when broken: **info**)
- **Why it matters:** Where the window runs past the record it is truncated (its weights stay put, the record's ends are tapered), so the spectra stay consistent -- but whatever arrives in the cut-off part (a pre-pulse, the start of the p/s pulse pair) is not measured, and the true resolution is set by the shorter, truncated record.
- **If it fails:** Start the scan earlier (or end it later) by the reported amount, or shrink config['preprocess']['window_half_width_ps'].

## Stage: `ellipsometry.calibration`

### the channel ratio C = d_p/d_s is frequency-flat

- **Check:** `channel_ratio_is_flat` (severity when broken: **warn**)
- **Why it matters:** By crystal symmetry the detection vector is real and frequency-independent, so structure in C means something else polarisation-dependent is in the beam: probe walk, astigmatism, a reference smaller or larger than the beam, polarising optics.
- **If it fails:** Look at the channel-ratio panel of the run figure. A slope suggests clipping or a reference/sample size mismatch; ripple suggests an echo inside the window.

### two independent calibration sources agree on the channel ratio when both can run

- **Check:** `calibration_sources_agree` (severity when broken: **warn**)
- **Why it matters:** Gold and probe rotation reach C by physically different routes -- one swaps an object into the focus, the other moves nothing in the THz path. Agreement validates both; disagreement localises an error to the geometry (gold) or the probe optics (rotation).
- **If it fails:** If they disagree, check the gold's tilt and size against the sample's, and the half-wave plate for wedge (plan sec. 7.8).

## Stage: `ellipsometry.inversion`

### the recovered index is passive (k >= 0) within its error bars

- **Check:** `index_is_passive` (severity when broken: **FAIL**)
- **Why it matters:** Under the package convention N = n - ik an absorbing sample has k >= 0. A k that is significantly negative is not physics: it is a sign-convention or calibration error, not a branch problem, because N follows linearly from (1 - rho)/(1 + rho).
- **If it fails:** Check the calibration source and its sign, and the emitter rotation sense; then the incidence angle.

### the sign convention is consistent end to end (no rho -> -rho flip)

- **Check:** `sign_convention_consistent` (severity when broken: **FAIL**)
- **Why it matters:** Mixing the Fresnel sign convention between the calibration and the inversion replaces rho with -rho. On silicon at 45 deg that returns eps ~ 0.52 instead of 11.7 (plan sec. 6.5) -- a number that looks like a material, not like an error.
- **If it fails:** Every model in the package uses thz_core's Fresnel coefficients; check any externally supplied (stored) channel ratio for its sign.

### the inversion is well conditioned across the band

- **Check:** `inversion_well_conditioned` (severity when broken: **info**)
- **Why it matters:** d eps/d rho grows as (1 + rho)^-3; for good conductors rho -> -1 and small errors in rho become large errors in eps (plan sec. 8.2). Bins there are noisy by physics, not by a fault, and should be read with their error bars.
- **If it fails:** Expected for metallic samples at 45 deg. A higher incidence angle improves it.

### the fitted out-of-plane tilt is small and determined

- **Check:** `tilt_is_plausible` (severity when broken: **warn**)
- **Why it matters:** A tilt of more than about a degree means the mount is far off; a tilt with an error bar as large as itself means the sample is too flat in frequency to separate tilt from the channel ratio, and the fit should not be trusted (HR-Si is the textbook case).
- **If it fails:** Re-seat the sample against the THz peak (plan sec. 4.5), or turn the tilt fit off for dispersionless samples.
