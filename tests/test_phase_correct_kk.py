"""Tests for the Kramers-Kronig (analytical fitting) phase correction.

Method: Jatkar, Yeh, Pancaldi & Bonetti, "Robust phase correction techniques for THz-TDS
in reflection" (arXiv:2412.18662). Core in thz_core.kramers_kronig; pipeline wrapper
thz.phase_correct_kk.

Known-answer strategy: plant a misplacement (a linear-in-omega phase = a pure time delay)
onto an intrinsic reflection coefficient, then check the method recovers the delay and
flattens the phase back to the intrinsic one — WITHOUT touching the amplitude.

Dependency-free; run with the project venv:
    .venv/Scripts/python.exe tests/test_phase_correct_kk.py
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thz_core.thz_core as core
from dataset_core.adapters import thz_adapter as thz

_C0 = 299_792_458.0


def _intrinsic_r(freq_hz):
    """A KK-consistent intrinsic reflection coefficient from a Drude dielectric.

    Using a physical (causal) model means the KK phase-from-amplitude relation holds, so
    the method attributes ~zero misplacement to it — a proper baseline for planting a
    known delay on top. A hand-built r with an arbitrary linear phase would NOT work: a
    linear phase IS a delay, and KK would (correctly) reinterpret it as misplacement.
    """
    omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    eps_inf, omega_p, gamma = 12.0, 2.0e12 * 2 * np.pi, 0.4e12 * 2 * np.pi
    eps = eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma + 1e-30)
    n = np.sqrt(eps)
    n = np.where(n.imag < 0, -n, n)
    r = (1.0 - n) / (1.0 + n)          # normal-incidence Fresnel (causal, minimum-phase)
    return r


def _plant_misplacement(freq_hz, r_intrinsic, delay_seconds):
    """Add phi = omega*delay (paper Eq. 2), i.e. multiply r by exp(-i*omega*delay)."""
    omega = 2.0 * np.pi * freq_hz
    return r_intrinsic * np.exp(-1j * omega * delay_seconds)


def _recovered_delay(freq, r, theta_rad=0.0, f_end=4.0e12):
    l, _ = core.estimate_misplacement(freq, r, theta_rad=theta_rad, f_end=f_end)
    return 2.0 * l / (_C0 * np.cos(theta_rad))


def test_core_recovers_planted_delay():
    freq = np.linspace(0.0, 6e12, 600)         # uniform, from ~0
    r_intrinsic = _intrinsic_r(freq)
    planted_delay_s = 0.040e-12                 # 40 fs
    r_measured = _plant_misplacement(freq, r_intrinsic, planted_delay_s)

    # Difference method: the small finite-band baseline (recovered on the intrinsic alone)
    # cancels, isolating the response to the planted delay.
    baseline_delay = _recovered_delay(freq, r_intrinsic)
    measured_delay = _recovered_delay(freq, r_measured)
    assert abs((measured_delay - baseline_delay) - planted_delay_s) < 3e-15, (
        measured_delay, baseline_delay, planted_delay_s)

    r_corrected, _, _ = core.correct_reflection_phase(freq, r_measured, f_end=4.0e12)
    # Amplitude untouched.
    assert np.allclose(np.abs(r_corrected), np.abs(r_measured))
    # Corrected phase matches the intrinsic (both processed identically) up to a constant.
    r_intrinsic_corrected, _, _ = core.correct_reflection_phase(freq, r_intrinsic, f_end=4.0e12)
    band = (freq > 0.5e12) & (freq < 3.5e12)
    residual = (np.unwrap(np.angle(r_corrected)) - np.unwrap(np.angle(r_intrinsic_corrected)))[band]
    residual -= residual.mean()
    assert np.max(np.abs(residual)) < 0.05, np.max(np.abs(residual))


def test_core_sign_is_correct_not_flipped():
    """Regression for the ill-conditioned-fit bug: +delay must recover as +delay."""
    freq = np.linspace(0.0, 6e12, 600)
    r_intrinsic = _intrinsic_r(freq)
    baseline = _recovered_delay(freq, r_intrinsic)
    for planted in (30e-15, -30e-15):
        measured = _recovered_delay(freq, _plant_misplacement(freq, r_intrinsic, planted))
        recovered = measured - baseline
        assert np.sign(recovered) == np.sign(planted), (planted, recovered)
        assert abs(recovered - planted) < 3e-15, (planted, recovered)


def test_delay_recovery_is_angle_independent():
    """The applied correction depends on the phase only; theta merely rescales l."""
    freq = np.linspace(0.0, 6e12, 600)
    r_intrinsic = _intrinsic_r(freq)
    planted_delay_s = 0.030e-12
    r_measured = _plant_misplacement(freq, r_intrinsic, planted_delay_s)

    responses = []
    for theta_deg in (0.0, 45.0):
        theta = np.deg2rad(theta_deg)
        baseline = _recovered_delay(freq, r_intrinsic, theta_rad=theta)
        measured = _recovered_delay(freq, r_measured, theta_rad=theta)
        responses.append(measured - baseline)
    # Same recovered delay regardless of the geometry angle used.
    assert abs(responses[0] - responses[1]) < 1e-15, responses
    assert abs(responses[0] - planted_delay_s) < 3e-15, responses


class _FakeDataService(dict):
    """Minimal stand-in: dict of filename->obj plus is_reference()."""
    def __init__(self, items, references):
        super().__init__(items)
        self._references = set(references)

    def is_reference(self, filename):
        return filename in self._references


class _FakeSample:
    def __init__(self, processing_dict):
        self.processing_dict = processing_dict
        self.data = None


class _FakeDataset:
    def __init__(self, data, config):
        self.data = data
        self.config = config


def test_adapter_wrapper_corrects_transfer_H():
    freq = np.linspace(0.0, 6e12, 600)
    planted_delay_s = 0.050e-12
    H_measured = _plant_misplacement(freq, _intrinsic_r(freq), planted_delay_s)

    sample = _FakeSample({
        "fft_freq": freq,
        "transfer_H": H_measured,
        "snr_db": np.full(freq.shape, 30.0),
        "transfer_mask": (freq <= 4.0e12) & (freq > 0.2e12),
    })
    dataset = _FakeDataset(
        _FakeDataService({"sample.acc": sample, "gold.acc": _FakeSample({})},
                         references=["gold.acc"]),
        config={"geometry": {"theta_external_deg": 45.0},
                "phase_kk": {"use_snr_mask": True}},
    )

    thz.phase_correct_kk(dataset)
    pd = sample.processing_dict

    # Original stashed, transfer_H overwritten, metrics recorded.
    assert "transfer_H_pre_kk" in pd
    assert np.allclose(pd["transfer_H_pre_kk"], H_measured)

    # Recovered delay (minus the finite-band baseline) matches the planted one, right sign.
    baseline_delay = _recovered_delay(freq, _intrinsic_r(freq),
                                      theta_rad=np.deg2rad(45.0), f_end=4.0e12)
    recovered_delay_s = pd["phase_kk_metrics"]["delay_seconds"]
    assert abs((recovered_delay_s - baseline_delay) - planted_delay_s) < 4e-15, recovered_delay_s

    # Corrected |H| unchanged; phase matches the intrinsic (both processed) up to a constant.
    assert np.allclose(np.abs(pd["transfer_H"]), np.abs(H_measured))
    r_intrinsic_corrected, _, _ = core.correct_reflection_phase(
        freq, _intrinsic_r(freq), theta_rad=np.deg2rad(45.0), f_end=4.0e12)
    band = (freq > 0.5e12) & (freq < 3.5e12)
    diff = (np.unwrap(np.angle(pd["transfer_H"]))
            - np.unwrap(np.angle(r_intrinsic_corrected)))[band]
    diff -= diff.mean()
    assert np.max(np.abs(diff)) < 0.05, np.max(np.abs(diff))


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {error}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
