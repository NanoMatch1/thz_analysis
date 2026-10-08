"""Time traces to spectra: baseline, one common window, zero pad, FFT -- and the resolution.

The arithmetic is ``thz_core.conditioning``, shared with the transmission and reflection
pipelines; the DECISIONS below are this pipeline's own.

The window is a fixed-width window applied at a COMMON centre for every acquisition, found from
the mean absolute trace. Using one centre for the whole series matters: windowing each trace at
its own peak would silently remove the very timing differences between polarisation settings that
the drift model exists to measure. The window also has to be wide enough to hold both the p and
the s pulse (their shapes differ because r_p != r_s) and short enough to exclude the echoes from
the emitter, the crystal and the sample's back face (plan sec. 6.2).

The window SHAPE matters for the drift model. A timing drift is a pure phase ramp only if the
window is flat where the pulses sit: under a sloped window a pulse that drifts also changes
amplitude, by (d ln w/dt) x drift. For a Hann window that is zero only at the exact centre, so any
second pulse -- the s pulse delayed against the p pulse, a background, a pre-pulse -- picks up a
drift-correlated amplitude error (0.7% for 30 fs at 0.4 ps off-centre in a 3 ps half-width,
against a 0.2% noise floor; caught by the harmonic chi-square). The default is therefore a
flat-top Tukey window (``taper_fraction`` = Tukey alpha, the part of the width given to the
cosine tapers), flat over the central part; 'hann' is kept for comparison with earlier runs.

Where the window runs past the start or end of the record (a short pre-pulse record is normal)
it is TRUNCATED, not resized: the weights stay where they are relative to the centre, so the flat
top stays on the pulse. The record's own edges are then half-cosine tapered over
``edge_taper_ps`` so the truncated window leaves no step for the FFT's wrap-around to see. How
much was cut is reported (``window_placement.clipped_before/after``).

The exact weighting used (window x edge taper) is returned, because the noise model must
propagate through the SAME weighting or its error bars describe a different spectrum. So is the
instrument resolution: zero padding interpolates, it does not add information, and a fit or a
plot that treats every padded bin as independent over-counts by ``resolution.oversampling``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from thz_core.thz_core import conditioning

__all__ = ["TransformedSeries", "spectra_from_traces", "transform_traces"]

#: The window shapes this pipeline offers; the shapes themselves are thz_core's.
WINDOW_SHAPE_NAMES = ("tukey", "hann")


@dataclass(frozen=True)
class TransformedSeries:
    frequencies_hz: np.ndarray      #: (n_frequencies,)
    spectra: np.ndarray             #: (n_acquisitions, n_frequencies) complex
    window: np.ndarray              #: (n_samples,) the total weighting applied, ones if none
    padded_length: int              #: FFT length
    time_step_s: float
    #: where the fixed window sits and how much of it the record cut off; None without a window
    window_placement: conditioning.FixedWidthWindow | None = None
    #: true resolution of the weighted record against the FFT bin spacing
    resolution: conditioning.InstrumentResolution | None = None


def transform_traces(time_ps, traces, *, window_half_width_ps=None, baseline_fraction=0.1,
                     pad_factor=4, window_centre_ps=None, window_shape="tukey",
                     taper_fraction=0.5, edge_taper_ps=0.5):
    """Baseline-subtract, window and transform a set of traces. Returns TransformedSeries.

    ``taper_fraction`` is the part of the Tukey window's full width given to the two cosine
    tapers; 0.5 leaves the central half flat. ``edge_taper_ps`` is the half-cosine ramp put on
    the record's own ends when a window is used (see the module docstring).
    """
    if window_shape not in WINDOW_SHAPE_NAMES:
        raise ValueError(f"unknown window_shape {window_shape!r}; known: "
                         f"{sorted(WINDOW_SHAPE_NAMES)}")
    time_ps = np.asarray(time_ps, dtype=float)
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    sample_count = time_ps.size
    if traces.shape[1] != sample_count:
        raise ValueError(f"traces have {traces.shape[1]} samples but the time axis has "
                         f"{sample_count}")

    corrected = conditioning.baseline_from_leading_samples(
        traces, max(int(baseline_fraction * sample_count), 1))

    if window_centre_ps is None:
        centre_index = int(np.argmax(np.abs(corrected).mean(axis=0)))
    else:
        centre_index = int(np.argmin(np.abs(time_ps - window_centre_ps)))

    time_step_ps = float(np.mean(np.diff(time_ps)))
    placement = None
    if window_half_width_ps is None:
        window = np.ones(sample_count)
    else:
        half_width = max(int(round(window_half_width_ps / time_step_ps)), 2)
        placement = conditioning.fixed_width_window(sample_count, centre_index, half_width,
                                                    window_shape, taper_fraction)
        window, _ = conditioning.taper_measured_block_edges(
            placement.weights, np.ones(sample_count, dtype=bool),
            int(round(edge_taper_ps / time_step_ps)))

    padded_length = int(sample_count * max(pad_factor, 1))
    time_step_s = time_step_ps * 1e-12
    spectra = np.fft.rfft(corrected * window[None, :], n=padded_length, axis=1)
    frequencies_hz = np.fft.rfftfreq(padded_length, d=time_step_s)
    record_duration_s = conditioning.support_duration(time_ps * 1e-12, window)
    resolution = (None if record_duration_s is None else conditioning.instrument_resolution(
        record_duration_s, frequencies_hz[1]))
    return TransformedSeries(frequencies_hz=frequencies_hz, spectra=spectra, window=window,
                             padded_length=padded_length, time_step_s=time_step_s,
                             window_placement=placement, resolution=resolution)


def spectra_from_traces(time_ps, traces, **kwargs):
    """``(frequencies_hz, spectra)`` -- the two arrays most callers need."""
    transformed = transform_traces(time_ps, traces, **kwargs)
    return transformed.frequencies_hz, transformed.spectra
