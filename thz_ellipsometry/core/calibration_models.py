"""How a measured channel ratio C(f) becomes the C(f) that is divided out: a registry of models.

A calibration SOURCE (``calibration_sources``) says where the per-frequency channel ratio
C(f) = d_p/d_s comes from -- gold in the focus, the sample at two probe settings, a stored value.
A calibration MODEL, here, says what is believed about its frequency dependence, and so how the
noisy per-frequency measurement is reduced to the curve actually applied to the sample.

    constant                        C(f) = C0                                (the original design)
    constant_plus_delay             C(f) = C0 exp(-i 2 pi (f - f0) tau)
    constant_plus_delay_and_slope   C(f) = C0 [1 + s (f - f0)] exp(-i 2 pi (f - f0) tau)
    per_frequency                   C(f) = the measurement itself, bin by bin

Why a delay: the original design took the detection vector as real and frequency-flat by crystal
symmetry, which is true of the GaP response. But C is everything that differs between the p and
the s states apart from the sample, and a p/s ARRIVAL-TIME difference -- the two magnet states
reaching the detector a few fs apart -- is a phase linear in frequency. Measured on gold it is
about 4 fs (F46): +/-1.5 deg across the band, which the constant model leaves inside rho.

``tau`` uses the figures' convention: positive means the p channel arrives LATER than s, which
with numpy's FFT is a factor exp(-i 2 pi f tau). ``f0`` is the (weighted) centre of the band, so
C0 is the value there and is nearly uncorrelated with tau and s; ``s`` is the fractional change
of |C| per Hz (reported per THz).

The parametric models fit complex C(f) directly (not unwrapped phase), weighted by the measured
per-frequency variance when the source supplies one, so the noisy band edges count for less.
What a model does not describe stays visible as its residual in the calibration figure; it is not
silently passed on to the sample, which is the case against ``per_frequency``: that carries
everything specific to the reference measurement (its clipping, its ripple) onto the sample as if
it were the instrument.

Adding a model means defining it here with ``@calibration_model`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "CALIBRATION_MODELS",
    "CalibrationModel",
    "CalibrationModelFit",
    "calibration_model",
    "calibration_model_names",
    "fit_calibration_model",
]


@dataclass(frozen=True)
class CalibrationModelFit:
    """A model fitted to the measured C(f) over the band."""

    applied_per_frequency: np.ndarray        #: (n_frequencies,) complex, full axis
    value_at_reference: complex              #: C at the reference frequency (the summary value)
    reference_frequency_hz: float
    parameters: dict = field(default_factory=dict)        #: name -> float, readable units
    standard_errors: dict = field(default_factory=dict)   #: name -> float, same units
    standard_error: float = 0.0              #: |standard error| of value_at_reference


@dataclass(frozen=True)
class CalibrationModel:
    name: str
    function: object      #: (frequencies_hz, measured, band, variance) -> CalibrationModelFit
    summary: str


#: name -> CalibrationModel. The one place a model exists.
CALIBRATION_MODELS: dict[str, CalibrationModel] = {}


def calibration_model(function):
    """Register a calibration model; its docstring's first line is its description."""
    summary = (function.__doc__ or "").strip().splitlines()
    CALIBRATION_MODELS[function.__name__] = CalibrationModel(
        name=function.__name__, function=function, summary=summary[0] if summary else "")
    return function


def calibration_model_names():
    return sorted(CALIBRATION_MODELS)


def fit_calibration_model(name, frequencies_hz, measured_per_frequency, band,
                          variance_per_frequency=None):
    """Fit the registered model ``name`` to the measured C(f) over ``band``."""
    try:
        entry = CALIBRATION_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown calibration model {name!r}; registered: "
                       f"{calibration_model_names()}") from None
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    measured = np.asarray(measured_per_frequency, dtype=complex)
    band = (np.ones(frequencies_hz.shape, dtype=bool) if band is None
            else np.asarray(band, dtype=bool))
    if not band.any():
        raise ValueError("the band selects no frequencies for the calibration model")
    variance = (None if variance_per_frequency is None
                else np.asarray(variance_per_frequency, dtype=float))
    return entry.function(frequencies_hz, measured, band, variance)


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

def _band_centre_hz(frequencies_hz, weights):
    return float(np.sum(frequencies_hz * weights) / np.sum(weights))


def _weights(variance, band):
    """Per-band-bin least-squares weights 1/sigma^2, or uniform without a variance."""
    if variance is None:
        return np.ones(int(band.sum()))
    selected = variance[band]
    usable = np.isfinite(selected) & (selected > 0)
    if not usable.all():
        return np.ones(int(band.sum()))
    return 1.0 / selected


@calibration_model
def constant(frequencies_hz, measured, band, variance):
    """One complex constant, the band mean of C(f) -- the original, crystal-symmetry design."""
    selected = measured[band]
    mean = complex(np.mean(selected))
    centre = _band_centre_hz(frequencies_hz[band], np.ones(selected.size))
    return CalibrationModelFit(
        applied_per_frequency=np.full(frequencies_hz.shape, mean),
        value_at_reference=mean, reference_frequency_hz=centre,
        parameters={"magnitude": abs(mean), "phase_deg": float(np.rad2deg(np.angle(mean)))},
        standard_error=float(np.std(selected) / np.sqrt(selected.size)))


def _delay_slope_model(frequencies_hz, centre_hz, magnitude, phase_rad, delay_s, slope_per_hz):
    offset = frequencies_hz - centre_hz
    return (magnitude * (1.0 + slope_per_hz * offset)
            * np.exp(1j * (phase_rad - 2.0 * np.pi * offset * delay_s)))


def _fit_delay(frequencies_hz, measured, band, variance, with_slope):
    """Shared least-squares fit of C0 * [1 + s (f - f0)] * exp(-i 2 pi (f - f0) tau)."""
    band_frequencies = frequencies_hz[band]
    selected = measured[band]
    weights = _weights(variance, band)
    centre = _band_centre_hz(band_frequencies, weights)
    offset = band_frequencies - centre

    # Start from a straight line through the unwrapped phase: good enough to land in the right
    # 2 pi basin, after which the complex fit owns the answer.
    phase = np.unwrap(np.angle(selected))
    slope, intercept = np.polyfit(2.0 * np.pi * offset, phase, 1, w=np.sqrt(weights))
    start = [float(np.mean(np.abs(selected))), float(intercept), float(-slope) * 1e15]
    if with_slope:
        start.append(0.0)
    root_weights = np.sqrt(weights)

    def residual(vector):
        magnitude, phase_rad, delay_fs = vector[:3]
        slope_per_thz = vector[3] if with_slope else 0.0
        model = _delay_slope_model(band_frequencies, centre, magnitude, phase_rad,
                                   delay_fs * 1e-15, slope_per_thz * 1e-12)
        difference = (model - selected) * root_weights
        return np.concatenate([difference.real, difference.imag])

    result = least_squares(residual, start, xtol=1e-14, ftol=1e-14, x_scale="jac")
    jacobian = result.jac
    degrees_of_freedom = max(result.fun.size - result.x.size, 1)
    # With a measured variance the residuals are in sigma units, and their reduced chi-square
    # still scales the errors: the per-bin variance is over-sampled-correlated and may be off,
    # so the scatter about the model is the honest judge either way.
    scale = float(np.sum(result.fun**2) / degrees_of_freedom)
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * scale
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    except np.linalg.LinAlgError:
        errors = np.full(result.x.size, np.inf)

    magnitude, phase_rad, delay_fs = (float(value) for value in result.x[:3])
    slope_per_thz = float(result.x[3]) if with_slope else 0.0
    value = complex(magnitude * np.exp(1j * phase_rad))
    parameters = {"magnitude": magnitude, "phase_deg": float(np.rad2deg(phase_rad)),
                  "delay_fs": delay_fs}
    standard_errors = {"magnitude": float(errors[0]), "phase_deg": float(np.rad2deg(errors[1])),
                       "delay_fs": float(errors[2])}
    if with_slope:
        parameters["slope_percent_per_thz"] = 100.0 * slope_per_thz
        standard_errors["slope_percent_per_thz"] = 100.0 * float(errors[3])
    parameters["reference_frequency_thz"] = centre / 1e12
    applied = _delay_slope_model(frequencies_hz, centre, magnitude, phase_rad, delay_fs * 1e-15,
                                 slope_per_thz * 1e-12)
    return CalibrationModelFit(
        applied_per_frequency=applied, value_at_reference=value, reference_frequency_hz=centre,
        parameters=parameters, standard_errors=standard_errors,
        standard_error=float(np.hypot(errors[0], magnitude * errors[1])))


@calibration_model
def constant_plus_delay(frequencies_hz, measured, band, variance):
    """A constant plus a p/s delay: C0 exp(-i 2 pi (f - f0) tau), fitted over the band."""
    return _fit_delay(frequencies_hz, measured, band, variance, with_slope=False)


@calibration_model
def constant_plus_delay_and_slope(frequencies_hz, measured, band, variance):
    """A constant, a p/s delay and a linear |C| slope: C0 [1 + s (f - f0)] exp(-i 2 pi (f - f0) tau)."""
    return _fit_delay(frequencies_hz, measured, band, variance, with_slope=True)


@calibration_model
def per_frequency(frequencies_hz, measured, band, variance):
    """The measured C(f) itself, bin by bin: carries the reference's noise and quirks onto the sample."""
    selected = measured[band]
    weights = _weights(variance, band)
    centre = _band_centre_hz(frequencies_hz[band], weights)
    at_centre = complex(measured[int(np.argmin(np.abs(frequencies_hz - centre)))])
    error = (0.0 if variance is None
             else float(np.sqrt(np.median(variance[band]))))
    return CalibrationModelFit(
        applied_per_frequency=measured.copy(), value_at_reference=at_centre,
        reference_frequency_hz=centre,
        parameters={"magnitude": abs(at_centre),
                    "phase_deg": float(np.rad2deg(np.angle(at_centre))),
                    "bins": int(selected.size)},
        standard_error=error)
