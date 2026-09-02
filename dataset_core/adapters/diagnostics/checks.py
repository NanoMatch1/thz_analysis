"""The registered assumptions.

Each entry here was earned: it encodes something that actually went wrong, silently,
and cost time to find. Adding one is a single ``@diagnostic`` registration — the
runtime check and the ledger entry come from the same declaration.

Checks must be cheap and must not mutate the dataset. They read what the pipeline has
already computed.
"""

from __future__ import annotations

import numpy as np

from .registry import Problem, Severity, diagnostic

_HZ_TO_THZ = 1e-12


def _samples(dataset):
    """(filename, data_obj) for every object, references included."""
    return list(dataset.data.items())


# ─────────────────────────────────────────────────────────────────────────────
# acquisition — is there anything to measure noise from?
# ─────────────────────────────────────────────────────────────────────────────


@diagnostic(
    stage="preprocessing",
    assumption="the individual scans behind each averaged trace are still available",
    why=(
        "Every repeat-based estimate — noise amplitudes, drift, spectral error bars — "
        "reads the per-scan matrix. When a row-count-changing step drops it, the matrix "
        "silently becomes the averaged trace as a single 'scan', whose scatter is zero. "
        "Nothing errors; the estimates just quietly describe nothing. This is how it was "
        "being lost by the very first pipeline step."
    ),
    remedy=(
        "the step that changed the row count must call thz_adapter.carry_scan_matrix "
        "with its own transform; until then, treat any repeat-based number for this "
        "file as unavailable rather than as a small value"
    ),
    severity=Severity.FAIL,
)
def per_scan_data_survived(dataset, config):
    from dataset_core.adapters.thz_adapter import scan_matrix_status

    for filename, data_obj in _samples(dataset):
        status = scan_matrix_status(data_obj)
        if status["reason"] is not None:
            yield Problem(
                filename=filename,
                message=f"per-scan data was discarded during processing "
                        f"({status['reason']})",
                detail=status,
            )


@diagnostic(
    stage="noise",
    assumption="there are enough repeats for the scatter between them to mean anything",
    why=(
        "A variance estimated from M samples carries a relative error of 1/sqrt(2(M-1)) "
        "— about 40% at M=4, 13% at M=30. Below roughly eight repeats the noise estimate "
        "is itself noise. Note this depends only on the NUMBER of repeats, not on how "
        "noisy each one is: a single scan being too weak to interpret alone is not an "
        "obstacle, because the DFT is linear."
    ),
    remedy="acquire more scans, or quote the estimate with its own uncertainty attached",
    severity=Severity.WARN,
)
def enough_repeats_for_noise(dataset, config):
    from dataset_core.adapters.thz_adapter import scan_matrix_status

    minimum = int((config.get("noise", {}) or {}).get("minimum_repeats", 8))
    for filename, data_obj in _samples(dataset):
        status = scan_matrix_status(data_obj)
        if status["reason"] is not None:
            continue  # already reported as a loss; not a count problem
        if 1 < status["n_scans"] < minimum:
            relative_error = 1.0 / np.sqrt(2 * (status["n_scans"] - 1))
            yield Problem(
                filename=filename,
                message=f"only {status['n_scans']} repeats; a noise estimate from these "
                        f"carries about {relative_error:.0%} relative error",
                detail={"n_scans": status["n_scans"], "minimum": minimum},
            )


@diagnostic(
    stage="acquisition",
    assumption="the instrument held still over the acquisition",
    why=(
        "Amplitude and arrival time drift over a run — on our purge box, 5% and 9 fs "
        "over 80 minutes while the nitrogen equilibrates. Drift is not removed by "
        "averaging, and a measurement can be drift-limited rather than noise-limited, in "
        "which case acquiring more scans does nothing at all. It also inflates any naive "
        "repeat-scatter estimate that does not fit it out first."
    ),
    remedy=(
        "wait for the purge to settle, or interleave sample and reference A-B-A-B so the "
        "drift affects both equally; check drift_inflation before trusting more averaging"
    ),
    severity=Severity.WARN,
)
def acquisition_drift(dataset, config):
    from thz_core.thz_core.noise import drift_corrected_scatter
    from dataset_core.adapters.thz_adapter import _SCAN_MATRIX_KEY, scan_matrix_status

    threshold = float((config.get("noise", {}) or {}).get("drift_inflation_warn", 1.5))
    for filename, data_obj in _samples(dataset):
        if not scan_matrix_status(data_obj)["intact"]:
            continue
        matrix = np.asarray(data_obj.processing_dict[_SCAN_MATRIX_KEY], dtype=float)
        if matrix.shape[1] < 4:
            continue
        dt = float(np.median(np.diff(matrix[:, 0])))
        estimate = drift_corrected_scatter(matrix[:, 1:].T, dt)
        if estimate.drift_inflation > threshold:
            yield Problem(
                filename=filename,
                message=f"acquisition drifted: raw repeat scatter is "
                        f"{estimate.drift_inflation:.2f}x the drift-corrected scatter "
                        f"(amplitude {np.ptp(estimate.amplitudes) * 100:.1f}% p-p, "
                        f"delay {np.ptp(estimate.delays) * 1e15:.1f} fs p-p)",
                detail={"drift_inflation": float(estimate.drift_inflation)},
            )


@diagnostic(
    stage="acquisition",
    assumption="the purge has equilibrated before the acquisition started",
    why=(
        "Water vapour absorbs across the THz band, and its continuum absorption grows "
        "with frequency. So while a nitrogen purge is still displacing room air, the "
        "measured spectrum RISES, and rises MORE at high frequency. A measurement taken "
        "through that transient has a sample and a reference recorded in different "
        "atmospheres, which is a systematic no amount of averaging removes. "
        "\n\n"
        "Crucially this does NOT require resolving the water lines: unresolved is not "
        "invisible, because a line narrower than the resolution still removes its "
        "energy from the band. Measured on a 111-scan CNT acquisition over 83 minutes, "
        "the band gain went +2.0% at 0.4-1 THz, +7.0% at 2-3 THz, +12.9% at 3-4 THz and "
        "+17.1% at 4-6 THz, while a settled reference acquisition on the same instrument "
        "was flat. "
        "\n\n"
        "The TILT is what identifies water specifically: a laser power drift scales the "
        "whole spectrum uniformly and leaves the tilt unchanged, so a frequency-"
        "dependent gain cannot be explained that way."
    ),
    remedy=(
        "wait for the purge to settle (99% settling has been measured at ~3.7 hours, not "
        "the assumed 15 minutes), or interleave sample and reference A-B-A-B so both see "
        "the same atmosphere"
    ),
    severity=Severity.WARN,
)
def purge_still_equilibrating(dataset, config):
    from dataset_core.adapters.thz_adapter import _SCAN_MATRIX_KEY, scan_matrix_status

    purge_config = config.get("purge", {}) or {}
    low_band = tuple(purge_config.get("low_band_thz", (0.4, 1.0)))
    high_band = tuple(purge_config.get("high_band_thz", (2.0, 3.0)))
    tilt_limit = float(purge_config.get("tilt_change_limit", 0.02))
    minimum_snr = float(purge_config.get("minimum_band_snr", 5.0))

    for filename, data_obj in _samples(dataset):
        if not scan_matrix_status(data_obj)["intact"]:
            continue
        matrix = np.asarray(data_obj.processing_dict[_SCAN_MATRIX_KEY], dtype=float)
        waveforms = matrix[:, 1:].T
        if waveforms.shape[0] < 8:
            continue
        dt = float(np.median(np.diff(matrix[:, 0])))
        n_samples = waveforms.shape[1]
        spectra = np.abs(np.fft.rfft(waveforms * np.hanning(n_samples),
                                     n=n_samples, axis=1))
        frequency_thz = np.fft.rfftfreq(n_samples, dt) * _HZ_TO_THZ

        def band_mean(band):
            inside = (frequency_thz >= band[0]) & (frequency_thz < band[1])
            return spectra[:, inside].mean(axis=1) if inside.any() else None

        low = band_mean(low_band)
        high = band_mean(high_band)
        if low is None or high is None or not np.all(low > 0):
            continue

        # Only trust the high band if it is actually above the top-of-axis level.
        floor = spectra[:, frequency_thz > 0.75 * frequency_thz.max()].mean()
        if floor <= 0 or high.mean() / floor < minimum_snr:
            continue

        tilt = high / low
        edge = max(3, tilt.size // 10)
        tilt_change = float(np.mean(tilt[-edge:]) / np.mean(tilt[:edge]) - 1.0)
        gain_change = float(np.mean(high[-edge:]) / np.mean(high[:edge]) - 1.0)
        if abs(tilt_change) > tilt_limit:
            yield Problem(
                filename=filename,
                message=(
                    f"the spectrum is still changing shape across the acquisition: the "
                    f"{high_band[0]}-{high_band[1]} THz band moved {gain_change:+.1%} "
                    f"relative to itself and {tilt_change:+.1%} relative to "
                    f"{low_band[0]}-{low_band[1]} THz. A frequency-dependent drift like "
                    f"this is the purge still displacing water, not laser power"
                ),
                detail={"tilt_change": tilt_change, "gain_change": gain_change,
                        "low_band_thz": low_band, "high_band_thz": high_band},
            )


# ─────────────────────────────────────────────────────────────────────────────
# spectral — is the noise floor a noise floor?
# ─────────────────────────────────────────────────────────────────────────────


@diagnostic(
    stage="transfer_function",
    assumption="the band the noise floor is read from has reached a noise plateau",
    why=(
        "The tail-median floor is a DYNAMIC RANGE metric (Naftaly & Dudley 2009), and it "
        "only measures noise if the spectrum has flattened out in the band it is read "
        "from. On a short record with a sharp pulse it has not — real pulse content "
        "persists to the sampling limit — so the 'floor' is set by signal. Measured on "
        "our CNT data it sat 8-38x above the true random scatter, did NOT fall as "
        "1/sqrt(M) when more scans were averaged, and reported a session that was 2.3x "
        "quieter as 3.9x worse. Because the transfer error is built as floor/|Y|, it "
        "then grows toward high frequency for reasons unrelated to the measurement."
    ),
    remedy=(
        "estimate the uncertainty from the repeat scans instead (thz_core.noise); treat "
        "any floor-derived error bar as an upper bound in the meantime"
    ),
    severity=Severity.WARN,
)
def noise_floor_band_is_a_plateau(dataset, config):
    tail_fraction = float((config.get("mask", {}) or {}).get("tail_fraction", 0.2))
    decay_limit = float((config.get("mask", {}) or {}).get("plateau_decay_limit", 2.0))

    for filename, data_obj in _samples(dataset):
        spectrum = data_obj.processing_dict.get("fft_spectrum")
        if spectrum is None:
            continue
        magnitude = np.abs(np.asarray(spectrum))
        tail_points = max(8, int(np.ceil(magnitude.size * tail_fraction)))
        band = magnitude[-tail_points:]
        third = max(1, band.size // 3)
        leading, trailing = np.median(band[:third]), np.median(band[-third:])
        if trailing <= 0:
            continue
        decay = float(leading / trailing)
        if decay > decay_limit:
            yield Problem(
                filename=filename,
                message=f"the floor band is still falling ({decay:.1f}x across it), so "
                        f"the floor is signal-limited, not noise-limited",
                detail={"decay_ratio": decay, "tail_fraction": tail_fraction},
            )


@diagnostic(
    stage="window",
    assumption="the analysis window fits inside the recorded trace",
    why=(
        "A window wider than the record is clipped at the trace edge, so it does not "
        "return to zero and the effective apodisation is not the one requested. The "
        "record length also sets the true frequency resolution (1/T), so a clipped "
        "window means the resolution being quoted is not the resolution being measured."
    ),
    remedy="reduce half_width_ps, or acquire a longer trace",
    severity=Severity.WARN,
)
def window_fits_the_record(dataset, config):
    for filename, data_obj in _samples(dataset):
        plan = data_obj.processing_dict.get("fixed_window")
        if plan and plan.get("clipped"):
            yield Problem(
                filename=filename,
                message=f"the {plan['window_length']}-sample window was clipped by the "
                        f"trace edge",
                detail=dict(plan),
            )


@diagnostic(
    stage="resolution",
    assumption="plotted and stored spectra are on the resolution actually measured",
    why=(
        "Zero-padding to a large n_fft makes the BIN SPACING far finer than the "
        "RESOLUTION, which is fixed at 1/T by the windowed record length. The extra "
        "points are sinc interpolation, not measurement, and reading structure from them "
        "reads structure from the padding."
    ),
    remedy=(
        "markers are placed on the independent grid automatically; set "
        "config['resolution']['limit_to_instrument_resolution'] to decimate the stored "
        "arrays too"
    ),
    severity=Severity.INFO,
)
def spectra_are_oversampled(dataset, config):
    resolution = config.get("resolution", {}) or {}
    factor = int(resolution.get("decimation_factor", 1))
    if factor > 1 and not resolution.get("resolution_applied", False):
        yield Problem(
            message=f"spectra are {factor}x oversampled relative to the true "
                    f"{resolution.get('df_resolution_hz', float('nan')) * _HZ_TO_THZ:.3f} "
                    f"THz resolution; only every {factor}th point is independent",
            detail={"decimation_factor": factor},
        )
