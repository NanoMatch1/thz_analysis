"""Per-acquisition spectral noise from the repeat scans, via the repo's noise model.

Each acquisition holds a few repeat scans. ``thz_core.noise.drift_corrected_scatter`` measures
the time-domain noise amplitude sigma(t) across them with the amplitude and delay drift between
repeats removed inside the estimator, and ``spectral_noise_moments`` propagates it exactly
through the windowed DFT (``ANALYSIS_NOTES`` sec. 20). The result, one variance per acquisition
per frequency, weights the harmonic fit and becomes the error bars on n and k.

Why the non-parametric estimator and not the three-term ``fit_noise_parameters``: the module
itself names this one as the one that should feed error bars, and on our repeat counts it is the
honest one -- checked on pure additive noise, its variance comes back at 0.98-1.00 of the truth
for 3-8 repeats, whereas the three-term fit returned sigma_alpha at 0.30x (4 repeats) and 0.65x
(8 repeats) of the truth, which would have shrunk every error bar.

It must be propagated through the SAME window the spectra were made with, or the bars describe a
different spectrum -- so this takes the ``TransformedSeries`` the preprocessing returned.

When there are too few repeats for the fit to mean anything, this returns None and the fit falls
back to residual-scatter errors. It never mixes weighted and unweighted acquisitions in one fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from thz_core.thz_core import noise as core_noise

__all__ = ["SeriesNoise", "estimate_series_noise"]


@dataclass(frozen=True)
class SeriesNoise:
    spectral_variance: np.ndarray | None    #: (n_acquisitions, n_frequencies) or None
    drift_estimates: tuple                  #: thz_core DriftEstimate per acquisition
    reason: str                             #: why it is None, or "ok"


def estimate_series_noise(series, transformed, *, minimum_scans=3):
    """E|dS|^2 per acquisition and frequency for a PolarisationSeries, or None with a reason."""
    if not series.files:
        return SeriesNoise(None, (), "the series carries no per-scan data")
    fewest = int(np.min(series.scan_counts))
    if fewest < minimum_scans:
        return SeriesNoise(None, (), f"only {fewest} repeat scans in the sparsest acquisition; "
                                     f"the noise model needs {minimum_scans}")

    time_step_s = transformed.time_step_s
    variances, estimates = [], []
    for file in series.files:
        try:
            estimate = core_noise.drift_corrected_scatter(file.scans, time_step_s)
        except Exception as error:  # noqa: BLE001 -- report and fall back, never halt
            return SeriesNoise(None, tuple(estimates),
                               f"noise estimate failed on {file.path}: {error}")
        moments = core_noise.spectral_noise_moments(
            estimate.sigma_t, window=transformed.window, n_fft=transformed.padded_length,
            n_averaged=file.scans.shape[0])
        variance = moments.variance_real + moments.variance_imag
        if not np.any(variance > 0.0):
            return SeriesNoise(None, tuple(estimates),
                               f"the repeat scans in {file.path} are identical (zero measured "
                               "noise -- noise-free synthetic data?); weights would be infinite")
        variances.append(variance)
        estimates.append(estimate)
    return SeriesNoise(np.vstack(variances), tuple(estimates), "ok")
