"""Time traces to spectra: baseline, one common window, zero pad, FFT.

The window is a fixed-width Hann applied at a COMMON centre for every acquisition, found from the
mean absolute trace. Using one centre for the whole series matters: windowing each trace at its
own peak would silently remove the very timing differences between polarisation settings that
the drift model exists to measure. The window also has to be wide enough to hold both the p and
the s pulse (their shapes differ because r_p != r_s) and short enough to exclude the echoes from
the emitter, the crystal and the sample's back face (plan sec. 6.2).

The window SHAPE matters for the drift model. A timing drift is a pure phase ramp only if the
window is flat where the pulses sit: under a sloped window a pulse that drifts also changes
amplitude, by (d ln w/dt) x drift. For a Hann window that is zero only at the exact centre, so any
second pulse -- the s pulse delayed against the p pulse, a background, a pre-pulse -- picks up a
drift-correlated amplitude error (0.7% for 30 fs at 0.4 ps off-centre in a 3 ps half-width,
against a 0.2% noise floor; caught by the harmonic chi-square). The default is therefore a
flat-top Tukey window, flat over the central part and tapered only at the edges; 'hann' is kept
for comparison with earlier runs.

The exact window used is returned, because the noise model must propagate through the SAME
window or its error bars describe a different spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WINDOW_SHAPES", "TransformedSeries", "spectra_from_traces", "transform_traces"]


def _tukey(length, taper_fraction):
    """Flat top with cosine tapers covering ``taper_fraction`` of the length (split both ends)."""
    if taper_fraction <= 0.0:
        return np.ones(length)
    if taper_fraction >= 1.0:
        return np.hanning(length)
    position = np.linspace(0.0, 1.0, length)
    window = np.ones(length)
    edge = taper_fraction / 2.0
    rising = position < edge
    falling = position > 1.0 - edge
    window[rising] = 0.5 * (1.0 - np.cos(np.pi * position[rising] / edge))
    window[falling] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - position[falling]) / edge))
    return window


#: name -> builder(length, taper_fraction). The one place a window shape exists.
WINDOW_SHAPES = {
    "tukey": _tukey,
    "hann": lambda length, taper_fraction: np.hanning(length),
}


@dataclass(frozen=True)
class TransformedSeries:
    frequencies_hz: np.ndarray      #: (n_frequencies,)
    spectra: np.ndarray             #: (n_acquisitions, n_frequencies) complex
    window: np.ndarray              #: (n_samples,) the window applied, ones if none
    padded_length: int              #: FFT length
    time_step_s: float


def transform_traces(time_ps, traces, *, window_half_width_ps=None, baseline_fraction=0.1,
                     pad_factor=4, window_centre_ps=None, window_shape="tukey",
                     taper_fraction=0.5):
    """Baseline-subtract, window and transform a set of traces. Returns TransformedSeries.

    ``taper_fraction`` is the part of the Tukey window's full width given to the two cosine
    tapers; 0.5 leaves the central half flat.
    """
    if window_shape not in WINDOW_SHAPES:
        raise ValueError(f"unknown window_shape {window_shape!r}; known: {sorted(WINDOW_SHAPES)}")
    time_ps = np.asarray(time_ps, dtype=float)
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    sample_count = time_ps.size
    if traces.shape[1] != sample_count:
        raise ValueError(f"traces have {traces.shape[1]} samples but the time axis has "
                         f"{sample_count}")

    baseline_count = max(int(baseline_fraction * sample_count), 1)
    corrected = traces - traces[:, :baseline_count].mean(axis=1, keepdims=True)

    if window_centre_ps is None:
        centre_index = int(np.argmax(np.abs(corrected).mean(axis=0)))
    else:
        centre_index = int(np.argmin(np.abs(time_ps - window_centre_ps)))

    time_step_ps = float(np.mean(np.diff(time_ps)))
    if window_half_width_ps is None:
        window = np.ones(sample_count)
    else:
        half_width = max(int(round(window_half_width_ps / time_step_ps)), 2)
        window = np.zeros(sample_count)
        start = max(centre_index - half_width, 0)
        stop = min(centre_index + half_width + 1, sample_count)
        window[start:stop] = WINDOW_SHAPES[window_shape](stop - start, taper_fraction)

    padded_length = int(sample_count * max(pad_factor, 1))
    time_step_s = time_step_ps * 1e-12
    spectra = np.fft.rfft(corrected * window[None, :], n=padded_length, axis=1)
    frequencies_hz = np.fft.rfftfreq(padded_length, d=time_step_s)
    return TransformedSeries(frequencies_hz=frequencies_hz, spectra=spectra, window=window,
                             padded_length=padded_length, time_step_s=time_step_s)


def spectra_from_traces(time_ps, traces, **kwargs):
    """``(frequencies_hz, spectra)`` -- the two arrays most callers need."""
    transformed = transform_traces(time_ps, traces, **kwargs)
    return transformed.frequencies_hz, transformed.spectra
