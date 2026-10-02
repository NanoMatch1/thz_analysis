"""Validation with explicit pass criteria, so "it worked" is a number and not an impression.

The gates implemented here are V1-V5 of the MVP plan: everything that can be checked against a
known answer without beam time. V6-V8 compare against real transmission measurements and are
driven from the run_me script once that data exists.

One result the criteria have to encode honestly: recovering ``k = 0`` on high-resistivity
silicon is the CORRECT answer. Its true k is about 1.2e-4 at 1 THz, some 250 times below the
measurement floor of a reflection ellipsometer at a realistic noise level, so a tolerance on k
for that sample is a tolerance on zero, not a sensitivity claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Check", "ValidationReport", "validate_index_against_reference"]


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    tolerance: float
    passed: bool
    detail: str = ""

    def __str__(self):
        mark = "PASS" if self.passed else "FAIL"
        suffix = f"  {self.detail}" if self.detail else ""
        return f"[{mark}] {self.name}: {self.value:.5g} (tolerance {self.tolerance:.5g}){suffix}"


@dataclass
class ValidationReport:
    label: str
    checks: list = field(default_factory=list)

    def add(self, name, value, tolerance, detail=""):
        self.checks.append(Check(name=name, value=float(value), tolerance=float(tolerance),
                                 passed=bool(abs(value) <= tolerance), detail=detail))
        return self.checks[-1]

    @property
    def passed(self):
        return all(check.passed for check in self.checks)

    def __str__(self):
        header = f"=== validation: {self.label} === {'PASS' if self.passed else 'FAIL'}"
        return "\n".join([header] + [f"   {check}" for check in self.checks])


def validate_index_against_reference(result, expected_index, *, label="sample",
                                     tolerance_n=0.02, tolerance_k=0.05,
                                     tolerance_harmonic_residual=0.05,
                                     tolerance_channel_scatter=0.05,
                                     band_edge_allowance=3.0,
                                     expected_angle_deg=None, tolerance_angle_deg=0.1):
    """Compare a recovered index against a known one, with the quality flags included.

    ``expected_index`` may be a scalar or an array over the band actually inverted.
    """
    inversion = result.inversion
    expected = np.broadcast_to(np.asarray(expected_index, dtype=complex),
                               inversion.index.shape)

    report = ValidationReport(label=label)
    # Median first: it is the robust statistic and the one the pass/fail should turn on.
    # The maximum over the band is reported too, at a looser tolerance, because a single
    # noisy bin at a band edge should not fail an otherwise good measurement.
    report.add("median |n - n_expected|",
               float(np.median(np.abs(inversion.index.real - expected.real))),
               tolerance_n,
               detail=f"median n = {np.median(inversion.index.real):.4f}")
    report.add("median |k - k_expected|",
               float(np.median(np.abs(-inversion.index.imag - (-expected.imag)))),
               tolerance_k,
               detail=f"median k = {np.median(-inversion.index.imag):.5f}")
    report.add("max |n - n_expected|",
               float(np.max(np.abs(inversion.index.real - expected.real))),
               tolerance_n * band_edge_allowance,
               detail=f"band-edge allowance x{band_edge_allowance:g}")
    report.add("max |k - k_expected|",
               float(np.max(np.abs(-inversion.index.imag - (-expected.imag)))),
               tolerance_k * band_edge_allowance,
               detail=f"band-edge allowance x{band_edge_allowance:g}")
    report.add("harmonic residual (sample)", result.sample_fit.residual_norm,
               tolerance_harmonic_residual,
               detail=f"drift {np.ptp(result.sample_fit.delays_s) * 1e15:.1f} fs")
    if result.calibration is not None:
        report.add("channel ratio scatter", result.calibration.relative_scatter,
                   tolerance_channel_scatter,
                   detail=f"ratio = {result.calibration.ratio:.4f}")
    if expected_angle_deg is not None and result.incidence_angle_fit is not None:
        report.add("incidence angle discrepancy [deg]",
                   result.incidence_angle_fit.angle_deg - expected_angle_deg,
                   tolerance_angle_deg,
                   detail=f"fitted {result.incidence_angle_fit.angle_deg:.3f} deg")
    return report


def extinction_measurement_floor(field_noise_fraction, incidence_angle_rad, index_sample,
                                 index_incident=1.0):
    """Smallest k distinguishable from zero, given an additive field-noise floor.

    Used to state honestly whether a null result on k means "no absorption" or "below the
    floor". The noise enters on the two FIELDS, not on rho -- treating it as a relative error
    on rho makes the Brewster angle look spuriously accurate, because a multiplicative error on
    a vanishing r_p vanishes with it.
    """
    from .model import index_from_ellipsometric_ratio, reflection_coefficients

    reflection_p, reflection_s = reflection_coefficients(
        index_sample, incidence_angle_rad, index_incident)
    noise = field_noise_fraction * abs(reflection_s)
    worst = 0.0
    for perturbation in (noise, -noise, 1j * noise, -1j * noise):
        for perturbed_ratio in ((reflection_p + perturbation) / reflection_s,
                                reflection_p / (reflection_s + perturbation)):
            recovered = index_from_ellipsometric_ratio(
                perturbed_ratio, incidence_angle_rad, index_incident,
                reference_index=index_sample)
            worst = max(worst, abs(-recovered.imag - (-np.imag(index_sample))))
    return float(worst)
