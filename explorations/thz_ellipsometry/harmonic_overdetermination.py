"""Does the GaN paper's rotating-analyser trick transfer to our emitter-rotation design?

Agulto et al., Sci. Rep. 11, 18129 (2021) measure the reflected waveform at 24 analyser
angles instead of just the two that p and s need, fit the three-parameter harmonic that the
signal MUST follow, and read a per-waveform timing jitter out of the residual.  They report
a tenfold improvement in the standard deviation of the ellipsometric parameters *at equal
total measurement time* -- which means it cannot be averaging, since averaging at fixed time
buys nothing.  It is the systematic correction.

Our instrument has the same redundancy available by rotating the EMITTER polarisation, which
moves no optic at all.  This module asks three questions:

  1. Is the harmonic form actually over-determined enough to separate a per-acquisition delay
     from the sample response?
  2. How much does it buy for OUR sample and geometry, at equal total measurement time?
  3. Does it matter which nuisance we fit -- delay, gain, or both?

Why this is worth the trouble even though rho cancels common-mode errors: a delay COMMON to
both polarisations cancels exactly, but a delay DIFFERENTIAL between the acquisitions at
different polarisation settings does not.  It enters rho as exp(i*omega*dt) and is therefore
a first-order error.  It is the one channel that ellipsometry does not kill for free.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ellipsometry_conditioning_analysis import (
    balanced_probe_angle_rad,
    electro_optic_detection_vector,
    ellipsometric_ratio,
    index_from_ellipsometric_ratio,
    sample_index,
)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def measured_amplitudes(emitter_angles_rad, channel_p, channel_s, delays_s, frequency_hz):
    """Complex amplitude at one frequency for each emitter polarisation setting.

    S_k = [P cos(alpha_k) + Q sin(alpha_k)] * exp(i*omega*dt_k)

    P and Q fold the sample reflection together with the (unknown, fixed) detection
    channel response.  Only their ratio is ever needed, which is the whole point.
    """
    angular_frequency = 2.0 * np.pi * frequency_hz
    harmonic = channel_p * np.cos(emitter_angles_rad) + channel_s * np.sin(emitter_angles_rad)
    return harmonic * np.exp(1j * angular_frequency * np.asarray(delays_s))


def _pack(channel_p, channel_s, delays_s):
    return np.concatenate([[channel_p.real, channel_p.imag, channel_s.real, channel_s.imag],
                           np.asarray(delays_s) * 1e15])


def _unpack(parameters, count):
    channel_p = parameters[0] + 1j * parameters[1]
    channel_s = parameters[2] + 1j * parameters[3]
    delays = np.zeros(count)
    delays[1:] = parameters[4:4 + count - 1] * 1e-15   # first delay pinned: gauge fixing
    return channel_p, channel_s, delays


def fit_harmonic(emitter_angles_rad, observed, frequency_hz, fit_delays):
    """Least-squares fit of the harmonic, optionally with a per-acquisition delay.

    The delay of the first setting is pinned to zero: a delay COMMON to every setting is
    degenerate with the overall phase of (P, Q) and is also exactly the thing that cancels
    in rho, so there is nothing to estimate and nothing to lose.
    """
    count = len(emitter_angles_rad)
    start = _pack(observed[0] + 1e-9, observed[-1] + 1e-9, np.zeros(count))
    if not fit_delays:
        start = start[:4]

    def residual(parameters):
        if fit_delays:
            channel_p, channel_s, delays = _unpack(parameters, count)
        else:
            channel_p = parameters[0] + 1j * parameters[1]
            channel_s = parameters[2] + 1j * parameters[3]
            delays = np.zeros(count)
        model = measured_amplitudes(emitter_angles_rad, channel_p, channel_s,
                                    delays, frequency_hz)
        difference = model - observed
        return np.concatenate([difference.real, difference.imag])

    result = least_squares(residual, start, xtol=1e-14, ftol=1e-14, method="lm")
    if fit_delays:
        channel_p, channel_s, _ = _unpack(result.x, count)
    else:
        channel_p = result.x[0] + 1j * result.x[1]
        channel_s = result.x[2] + 1j * result.x[3]
    return channel_p, channel_s, result


def harmonic_residual_norm(emitter_angles_rad, observed, frequency_hz):
    """Residual of the 2-parameter harmonic fit -- a run-time quality flag.

    The signal MUST be a pure first harmonic in the emitter angle.  Whatever is left over
    is instrument error, and its size is measurable without knowing what caused it.
    """
    _, _, result = fit_harmonic(emitter_angles_rad, observed, frequency_hz, fit_delays=False)
    return np.linalg.norm(result.fun) / np.linalg.norm(np.abs(observed))


# ---------------------------------------------------------------------------
# Monte Carlo comparison at equal total measurement time
# ---------------------------------------------------------------------------

SCHEMES = ("two_point", "harmonic_no_delay_fit", "harmonic_delay_fit")


def _scheme_angles(scheme, harmonic_points):
    if scheme == "two_point":
        return np.array([0.0, np.pi / 2])
    return np.linspace(0.0, np.pi, harmonic_points, endpoint=False)


def run_trial(scheme, index_sample, incidence_angle_rad, frequency_hz, detection_ratio,
              relative_noise_unit_time, jitter_std_s, generator, harmonic_points):
    """One realisation.  Noise scales with sqrt(settings) so total time is held equal."""
    angles = _scheme_angles(scheme, harmonic_points)
    count = len(angles)

    true_ratio = ellipsometric_ratio(1.0, index_sample, incidence_angle_rad)
    channel_s = 1.0 + 0.0j
    channel_p = true_ratio * detection_ratio

    delays = generator.normal(0.0, jitter_std_s, size=count)
    clean = measured_amplitudes(angles, channel_p, channel_s, delays, frequency_hz)

    noise_scale = relative_noise_unit_time * np.sqrt(count) * np.abs(clean).mean()
    observed = clean + noise_scale * (generator.normal(size=count)
                                      + 1j * generator.normal(size=count)) / np.sqrt(2)

    if scheme == "two_point":
        recovered_ratio = (observed[0] / observed[1]) / detection_ratio
    else:
        fitted_p, fitted_s, _ = fit_harmonic(angles, observed, frequency_hz,
                                             fit_delays=(scheme == "harmonic_delay_fit"))
        recovered_ratio = (fitted_p / fitted_s) / detection_ratio

    recovered_index = index_from_ellipsometric_ratio(recovered_ratio, 1.0, incidence_angle_rad)
    return abs(recovered_index - index_sample), abs(recovered_ratio - true_ratio)


def compare_schemes(frequency_hz=1.0e12, incidence_angle_deg=70.0, jitter_std_fs=2.0,
                    relative_noise_unit_time=0.005, trials=400, harmonic_points=12, seed=5):
    angle = np.deg2rad(incidence_angle_deg)
    index = sample_index(frequency_hz)
    detection = electro_optic_detection_vector(balanced_probe_angle_rad())
    detection_ratio = detection[0] / detection[1]

    print(f"\n[scheme comparison] {frequency_hz/1e12:.1f} THz, {incidence_angle_deg:.0f} deg, "
          f"|N| = {abs(index):.2f}, per-acquisition jitter {jitter_std_fs:.1f} fs rms,")
    print(f"   noise {relative_noise_unit_time:.1%} at full time, "
          f"{harmonic_points} emitter settings, EQUAL total measurement time, {trials} trials")
    print(f"   {'scheme':>24} {'median |dN|':>13} {'90th pct':>11} {'median |drho|':>15}")

    results = {}
    for scheme in SCHEMES:
        generator = np.random.default_rng(seed)
        errors = np.array([
            run_trial(scheme, index, angle, frequency_hz, detection_ratio,
                      relative_noise_unit_time, jitter_std_fs * 1e-15, generator,
                      harmonic_points)
            for _ in range(trials)])
        index_error, ratio_error = errors[:, 0], errors[:, 1]
        results[scheme] = np.median(index_error)
        print(f"   {scheme:>24} {np.median(index_error):>13.4f} "
              f"{np.percentile(index_error, 90):>11.4f} {np.median(ratio_error):>15.5f}")
    print(f"   -> delay-fitted harmonic vs two-point: "
          f"{results['two_point']/results['harmonic_delay_fit']:.1f}x better")
    print(f"   -> harmonic WITHOUT the delay fit vs two-point: "
          f"{results['two_point']/results['harmonic_no_delay_fit']:.1f}x "
          f"(this is the averaging-only part)")
    return results


def jitter_sweep(jitter_values_fs=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0), **kwargs):
    print("\n[jitter sweep] where the gain comes from, as a function of differential timing error")
    print(f"   {'jitter [fs]':>12} {'two-point':>12} {'harmonic':>12} {'harmonic+dt':>13} "
          f"{'gain':>8}")
    for jitter_fs in jitter_values_fs:
        quiet = {**kwargs, "jitter_std_fs": jitter_fs}
        results = _silent_compare(**quiet)
        print(f"   {jitter_fs:>12.1f} {results['two_point']:>12.4f} "
              f"{results['harmonic_no_delay_fit']:>12.4f} "
              f"{results['harmonic_delay_fit']:>13.4f} "
              f"{results['two_point']/results['harmonic_delay_fit']:>7.1f}x")


def _silent_compare(**kwargs):
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return compare_schemes(**kwargs)


def settings_sweep(points_values=(4, 6, 8, 12, 24), **kwargs):
    print("\n[settings sweep] how many emitter settings are worth taking "
          "(equal total time throughout)")
    print(f"   {'settings':>10} {'harmonic+dt |dN|':>18} {'vs two-point':>14}")
    for points in points_values:
        results = _silent_compare(**{**kwargs, "harmonic_points": points})
        print(f"   {points:>10} {results['harmonic_delay_fit']:>18.4f} "
              f"{results['two_point']/results['harmonic_delay_fit']:>13.1f}x")


def validate(verbose=True):
    """Known-answer checks before any number above is believed."""
    failures = []
    angle = np.deg2rad(70.0)
    index = sample_index(1.0e12)
    true_ratio = ellipsometric_ratio(1.0, index, angle)
    angles = np.linspace(0.0, np.pi, 12, endpoint=False)

    # Noiseless, jitter-free: the harmonic fit must be exact.
    clean = measured_amplitudes(angles, true_ratio, 1.0 + 0j, np.zeros(12), 1.0e12)
    fitted_p, fitted_s, _ = fit_harmonic(angles, clean, 1.0e12, fit_delays=False)
    if abs(fitted_p / fitted_s - true_ratio) > 1e-9:
        failures.append(f"noiseless harmonic fit failed: {fitted_p/fitted_s} vs {true_ratio}")

    # Noiseless WITH known jitter: the delay-fitting model must recover rho exactly.
    delays = np.array([0.0, 3e-15, -2e-15, 5e-15, -1e-15, 4e-15,
                       -3e-15, 1e-15, 2e-15, -4e-15, 0.5e-15, -1.5e-15])
    jittered = measured_amplitudes(angles, true_ratio, 1.0 + 0j, delays, 1.0e12)
    fitted_p, fitted_s, _ = fit_harmonic(angles, jittered, 1.0e12, fit_delays=True)
    if abs(fitted_p / fitted_s - true_ratio) > 1e-7:
        failures.append(f"delay-fitted recovery failed: {fitted_p/fitted_s} vs {true_ratio}")

    # A common delay must NOT disturb rho at all, with or without the delay fit.
    common = measured_amplitudes(angles, true_ratio, 1.0 + 0j,
                                 np.full(12, 7e-15), 1.0e12)
    fitted_p, fitted_s, _ = fit_harmonic(angles, common, 1.0e12, fit_delays=False)
    if abs(fitted_p / fitted_s - true_ratio) > 1e-9:
        failures.append("common delay leaked into rho")

    # The residual flag must be zero on clean data and non-zero on jittered data.
    if harmonic_residual_norm(angles, clean, 1.0e12) > 1e-9:
        failures.append("residual flag non-zero on clean data")
    if harmonic_residual_norm(angles, jittered, 1.0e12) < 1e-4:
        failures.append("residual flag blind to jitter")

    if verbose:
        print("validation:", "PASS (5 known-answer checks)" if not failures else "FAIL")
        for failure in failures:
            print("   ", failure)
    return failures




# ---------------------------------------------------------------------------
# Our actual error model: slow drift, not white jitter -- and acquisition ORDER
# ---------------------------------------------------------------------------

def acquisition_schedule(scheme, total_acquisitions, harmonic_points):
    """Emitter angle for each acquisition, in the order it is taken.

    'sequential' finishes one polarisation setting before starting the next, which is how a
    rotating-analyser instrument naturally runs.  'interleaved' cycles through the settings
    repeatedly, which converts a slow drift into a common-mode error between neighbouring
    acquisitions -- the same conclusion F31 reached for the bare|doped comparison.
    """
    if scheme.startswith("two_point"):
        angles = np.array([0.0, np.pi / 2])
    else:
        angles = np.linspace(0.0, np.pi, harmonic_points, endpoint=False)
    count = len(angles)
    repeats = max(total_acquisitions // count, 1)
    if "interleaved" in scheme:
        order = np.tile(np.arange(count), repeats)
    else:
        order = np.repeat(np.arange(count), repeats)
    return angles, order


def drifting_delays(order, drift_span_s, white_jitter_s, generator):
    """Delay of each acquisition: a linear drift across the run plus white jitter."""
    count = len(order)
    elapsed_fraction = np.arange(count) / max(count - 1, 1)
    return drift_span_s * elapsed_fraction + generator.normal(0.0, white_jitter_s, size=count)


def _stack_by_setting(order, values, setting_count):
    """Average the complex amplitudes acquired at each setting."""
    return np.array([values[order == setting].mean() for setting in range(setting_count)])


def run_ordered_trial(scheme, index_sample, incidence_angle_rad, frequency_hz, detection_ratio,
                      relative_noise_unit_time, drift_span_s, white_jitter_s, generator,
                      harmonic_points, total_acquisitions):
    angles, order = acquisition_schedule(scheme, total_acquisitions, harmonic_points)
    setting_count = len(angles)

    true_ratio = ellipsometric_ratio(1.0, index_sample, incidence_angle_rad)
    channel_s = 1.0 + 0.0j
    channel_p = true_ratio * detection_ratio

    delays = drifting_delays(order, drift_span_s, white_jitter_s, generator)
    per_acquisition = measured_amplitudes(angles[order], channel_p, channel_s,
                                          delays, frequency_hz)
    noise_scale = (relative_noise_unit_time * np.sqrt(len(order))
                   * np.abs(per_acquisition).mean())
    per_acquisition = per_acquisition + noise_scale * (
        generator.normal(size=len(order)) + 1j * generator.normal(size=len(order))) / np.sqrt(2)

    averaged = _stack_by_setting(order, per_acquisition, setting_count)

    if scheme.startswith("two_point"):
        recovered_ratio = (averaged[0] / averaged[1]) / detection_ratio
    elif scheme.endswith("drift_fit"):
        recovered_ratio = _fit_with_linear_drift(angles, averaged, frequency_hz) / detection_ratio
    elif scheme.endswith("delay_fit"):
        fitted_p, fitted_s, _ = fit_harmonic(angles, averaged, frequency_hz, fit_delays=True)
        recovered_ratio = (fitted_p / fitted_s) / detection_ratio
    else:
        fitted_p, fitted_s, _ = fit_harmonic(angles, averaged, frequency_hz, fit_delays=False)
        recovered_ratio = (fitted_p / fitted_s) / detection_ratio

    recovered_index = index_from_ellipsometric_ratio(recovered_ratio, 1.0, incidence_angle_rad)
    return abs(recovered_index - index_sample)


def _fit_with_linear_drift(angles, averaged, frequency_hz):
    """Harmonic fit with the delay constrained to ONE parameter: a linear ramp.

    Our timing error is a slow drift (F31 measured ~17 fs/hour), not independent
    per-acquisition jitter.  Modelling it with N-1 free delays spends degrees of freedom we
    do not need to spend; a single ramp captures it and keeps the statistical power.
    """
    angular_frequency = 2.0 * np.pi * frequency_hz
    ramp = np.arange(len(angles)) / max(len(angles) - 1, 1)

    def residual(parameters):
        channel_p = parameters[0] + 1j * parameters[1]
        channel_s = parameters[2] + 1j * parameters[3]
        delays = parameters[4] * 1e-15 * ramp
        model = ((channel_p * np.cos(angles) + channel_s * np.sin(angles))
                 * np.exp(1j * angular_frequency * delays))
        difference = model - averaged
        return np.concatenate([difference.real, difference.imag])

    start = np.array([averaged[0].real, averaged[0].imag,
                      averaged[-1].real, averaged[-1].imag, 0.0])
    result = least_squares(residual, start, xtol=1e-14, ftol=1e-14, method="lm")
    return (result.x[0] + 1j * result.x[1]) / (result.x[2] + 1j * result.x[3])


ORDERED_SCHEMES = (
    "two_point_sequential",
    "two_point_interleaved",
    "harmonic_sequential",
    "harmonic_sequential_delay_fit",
    "harmonic_sequential_drift_fit",
    "harmonic_interleaved",
)


def compare_ordered(frequency_hz=1.0e12, incidence_angle_deg=70.0, drift_span_fs=20.0,
                    white_jitter_fs=0.5, relative_noise_unit_time=0.005, trials=300,
                    harmonic_points=12, total_acquisitions=48, seed=17, quiet=False):
    angle = np.deg2rad(incidence_angle_deg)
    index = sample_index(frequency_hz)
    detection = electro_optic_detection_vector(balanced_probe_angle_rad())
    detection_ratio = detection[0] / detection[1]

    if not quiet:
        print(f"\n[acquisition order] {frequency_hz/1e12:.1f} THz, {incidence_angle_deg:.0f} deg, "
              f"drift {drift_span_fs:.0f} fs across the run, white jitter {white_jitter_fs:.1f} fs,")
        print(f"   noise {relative_noise_unit_time:.1%}, {total_acquisitions} acquisitions total "
              f"(equal time for every scheme), {trials} trials")
        print(f"   {'scheme':>32} {'median |dN|':>13} {'90th pct':>11}")

    results = {}
    for scheme in ORDERED_SCHEMES:
        generator = np.random.default_rng(seed)
        errors = np.array([
            run_ordered_trial(scheme, index, angle, frequency_hz, detection_ratio,
                              relative_noise_unit_time, drift_span_fs * 1e-15,
                              white_jitter_fs * 1e-15, generator, harmonic_points,
                              total_acquisitions)
            for _ in range(trials)])
        results[scheme] = np.median(errors)
        if not quiet:
            print(f"   {scheme:>32} {np.median(errors):>13.4f} "
                  f"{np.percentile(errors, 90):>11.4f}")
    return results


def drift_sweep(drift_values_fs=(0.0, 5.0, 20.0, 60.0, 200.0), **kwargs):
    print("\n[drift sweep] which strategy survives how much drift across a run")
    print(f"   {'drift [fs]':>11} {'2pt seq':>10} {'2pt intlv':>11} {'harm seq':>10} "
          f"{'harm+dt':>10} {'harm+ramp':>11} {'harm intlv':>11}")
    for drift_fs in drift_values_fs:
        results = compare_ordered(drift_span_fs=drift_fs, quiet=True, **kwargs)
        print(f"   {drift_fs:>11.0f} "
              f"{results['two_point_sequential']:>10.4f} "
              f"{results['two_point_interleaved']:>11.4f} "
              f"{results['harmonic_sequential']:>10.4f} "
              f"{results['harmonic_sequential_delay_fit']:>10.4f} "
              f"{results['harmonic_sequential_drift_fit']:>11.4f} "
              f"{results['harmonic_interleaved']:>11.4f}")


def main():
    validate()
    compare_schemes()
    jitter_sweep()
    settings_sweep()
    compare_ordered()
    drift_sweep()
    print("\ndone.")


if __name__ == "__main__":
    main()
