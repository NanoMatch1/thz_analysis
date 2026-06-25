"""Unit tests for the air-gap slider explorer's pure de-embed / conductivity functions.

These pin the MATH the interactive tool relies on, with no GUI and no pipeline:
  * single-gap de-embed is the exact inverse of the single-gap forward model (sigma_d = 0),
  * the full de-embed -> invert round-trip recovers planted n, k,
  * conductivity_from_nk matches thz_core.derive_eps_sigma exactly,
  * the Debye-Waller width factor behaves (1 at sigma=0, monotone down, boost lifts high-f mag).

Run with:
    .venv/Scripts/python.exe tests/test_air_gap_slider_explorer.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "explorations", "air_gap_cnt_reflection"))

import thz_core.thz_core as core
from air_gap_slider_explorer import (
    internal_and_gap_angles,
    gap_round_trip_phase,
    deembed_single_gap,
    debye_waller_magnitude,
    deembed_reflection,
    invert_air_incidence,
    conductivity_from_nk,
    samples_from_dataset,
    _short_sample_name,
)

_, GAP_ANGLE_RAD = internal_and_gap_angles()
FREQ_HZ = np.linspace(0.3e12, 2.5e12, 200)
FULL_MASK = np.ones(FREQ_HZ.size, dtype=bool)
R_FRONT = complex(core.fresnel_reflection_s(1.95, 1.0, internal_and_gap_angles()[0]))  # SiO2->air


def _single_gap_forward(r_back, gap_thickness_m):
    """Exact single-gap Fabry-Perot: r_meas = (r_f + r_b e^{-i2b})/(1 + r_f r_b e^{-i2b})."""
    bounce = np.exp(-1j * gap_round_trip_phase(FREQ_HZ, gap_thickness_m, GAP_ANGLE_RAD))
    return (R_FRONT + r_back * bounce) / (1.0 + R_FRONT * r_back * bounce)


def _planted_cnt_reflection(n_value, k_value):
    """r_back = r_{air->CNT} for a (flat, dispersionless) planted n, k at the gap angle."""
    n_hat = n_value - 1j * k_value  # thz_core convention n_hat = n - i k
    return np.asarray(core.fresnel_reflection_s(1.0, n_hat, GAP_ANGLE_RAD), dtype=complex) \
        * np.ones(FREQ_HZ.size)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_single_gap_deembed_is_exact_inverse():
    """sigma_d = 0: de-embedding the single-gap forward recovers r_back to ~1e-12."""
    r_back = _planted_cnt_reflection(3.0, 1.5)
    gap_m = 18e-6
    r_meas = _single_gap_forward(r_back, gap_m)
    recovered = deembed_reflection(r_meas, R_FRONT, FREQ_HZ, gap_m, 0.0, GAP_ANGLE_RAD)
    assert np.allclose(recovered, r_back, atol=1e-12)


def test_deembed_then_invert_recovers_planted_nk():
    """Full chain: plant n,k -> forward through a gap -> de-embed -> invert -> recover n,k."""
    n_true, k_true = 3.4, 0.8
    r_back = _planted_cnt_reflection(n_true, k_true)
    gap_m = 22e-6
    r_meas = _single_gap_forward(r_back, gap_m)
    r_back_recovered = deembed_reflection(r_meas, R_FRONT, FREQ_HZ, gap_m, 0.0, GAP_ANGLE_RAD)
    n, k = invert_air_incidence(FREQ_HZ, r_back_recovered, FULL_MASK, GAP_ANGLE_RAD)
    assert np.allclose(n, n_true, atol=1e-6)
    assert np.allclose(k, k_true, atol=1e-6)


def test_wrong_position_biases_nk_away_from_truth():
    """A de-embed at the WRONG gap position must NOT return the planted n,k (sensitivity)."""
    r_back = _planted_cnt_reflection(3.0, 0.5)
    gap_m = 20e-6
    r_meas = _single_gap_forward(r_back, gap_m)
    r_back_wrong = deembed_reflection(r_meas, R_FRONT, FREQ_HZ, gap_m + 8e-6, 0.0, GAP_ANGLE_RAD)
    n_wrong, _ = invert_air_incidence(FREQ_HZ, r_back_wrong, FULL_MASK, GAP_ANGLE_RAD)
    assert not np.allclose(n_wrong, 3.0, atol=0.05)


def test_conductivity_matches_thz_core():
    """conductivity_from_nk reproduces thz_core.derive_eps_sigma for several eps_inf."""
    n = 2.5 + 0.4 * np.sin(FREQ_HZ / 3e11)
    k = 1.2 + 0.3 * np.cos(FREQ_HZ / 4e11)
    for eps_inf in (1.0, 4.0, 11.7):
        ours = conductivity_from_nk(FREQ_HZ, n, k, eps_inf)
        _, sigma_core, _ = core.derive_eps_sigma(
            FREQ_HZ, n, k, {"derive": {"eps_background": eps_inf}},
        )
        assert np.allclose(ours, sigma_core, rtol=1e-9, atol=1e-9)


def test_sigma_real_independent_of_eps_inf():
    """Re(sigma) = omega eps0 2 n k must not depend on eps_inf (only Im does)."""
    n = 3.0 * np.ones(FREQ_HZ.size)
    k = 1.0 * np.ones(FREQ_HZ.size)
    sigma_a = conductivity_from_nk(FREQ_HZ, n, k, 1.0)
    sigma_b = conductivity_from_nk(FREQ_HZ, n, k, 9.0)
    assert np.allclose(np.real(sigma_a), np.real(sigma_b), rtol=1e-12)
    assert not np.allclose(np.imag(sigma_a), np.imag(sigma_b))


def test_debye_waller_is_unity_at_zero_and_monotone_down():
    assert np.allclose(debye_waller_magnitude(FREQ_HZ, 0.0, GAP_ANGLE_RAD), 1.0)
    w_small = debye_waller_magnitude(FREQ_HZ, 4e-6, GAP_ANGLE_RAD)
    w_large = debye_waller_magnitude(FREQ_HZ, 9e-6, GAP_ANGLE_RAD)
    assert np.all(w_small <= 1.0 + 1e-12)
    assert np.all(w_large <= w_small + 1e-12)            # rougher -> more suppression
    assert w_large[-1] < w_large[0]                       # suppression grows with frequency


def test_width_correction_boosts_high_frequency_magnitude():
    """Undoing Debye-Waller lifts |r_back| at high f relative to no width correction."""
    r_back = _planted_cnt_reflection(3.0, 1.0)
    r_meas = _single_gap_forward(r_back, 18e-6)
    no_width = deembed_reflection(r_meas, R_FRONT, FREQ_HZ, 18e-6, 6e-6 * 0.0, GAP_ANGLE_RAD)
    with_width = deembed_reflection(r_meas, R_FRONT, FREQ_HZ, 18e-6, 6e-6, GAP_ANGLE_RAD)
    lift = np.abs(with_width) / np.abs(no_width)
    assert lift[-1] > lift[0]            # the magnitude boost grows with frequency
    assert lift[-1] > 1.05               # appreciable lift at the top of the band
    assert lift[0] < 1.01                # negligible lift at the bottom of the band


class _FakeDataAccess:
    def __init__(self, items, references):
        self._items = items
        self._references = set(references)

    def items(self):
        return list(self._items.items())

    def is_reference(self, name):
        return name in self._references


class _FakeObj:
    def __init__(self, processing):
        self.processing_dict = processing


class _FakeDataset:
    def __init__(self, items, references):
        self.data = _FakeDataAccess(items, references)


def _processed_sample(n_value=3.0):
    return _FakeObj({
        "fft_freq": FREQ_HZ,
        "transfer_mask": FULL_MASK,
        "reflection_r": _planted_cnt_reflection(n_value, 0.5),
        "r_reference": R_FRONT,
        "n": n_value * np.ones(FREQ_HZ.size),
        "k": 0.5 * np.ones(FREQ_HZ.size),
    })


def test_short_sample_name_handles_si_and_cnt():
    assert _short_sample_name("sample_CNT-0-deg_camera-align-T0.acc") == "CNT-0-deg"
    assert _short_sample_name("sample_Si-0-deg_camera-align-T0.acc") == "Si-0-deg"
    assert _short_sample_name("reference_silicon_x.acc") == "silicon"


def test_samples_from_dataset_extracts_records():
    """Builds one record per non-reference sample with the keys the de-embed needs."""
    ds = _FakeDataset(
        {"sample_CNT-0-deg_x.acc": _processed_sample(3.0),
         "reference_sio2_x.acc": _FakeObj({})},  # reference skipped before processing read
        references={"reference_sio2_x.acc"},
    )
    samples = samples_from_dataset(ds)
    assert len(samples) == 1
    record = samples[0]
    assert record["name"] == "CNT-0-deg"
    assert record["frequency_hz"].shape == FREQ_HZ.shape
    assert record["r_meas"].dtype == complex
    assert isinstance(record["r_front"], complex)
    # the extracted record must drive the de-embed chain without error
    n, k = invert_air_incidence(record["frequency_hz"],
                                deembed_reflection(record["r_meas"], record["r_front"],
                                                   record["frequency_hz"], 0.0, 0.0, GAP_ANGLE_RAD),
                                record["mask"], GAP_ANGLE_RAD)
    assert np.all(np.isfinite(n))


def test_samples_from_dataset_raises_when_no_inverted_samples():
    """A dataset without reflection_r (inversion not run) is a clear error, not empty silence."""
    ds = _FakeDataset(
        {"sample_x.acc": _FakeObj({"fft_freq": FREQ_HZ})},  # no reflection_r
        references=set(),
    )
    raised = False
    try:
        samples_from_dataset(ds)
    except RuntimeError as error:
        raised = True
        assert "invert_nk_reflection" in str(error)
    assert raised


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_single_gap_deembed_is_exact_inverse,
    test_deembed_then_invert_recovers_planted_nk,
    test_wrong_position_biases_nk_away_from_truth,
    test_conductivity_matches_thz_core,
    test_sigma_real_independent_of_eps_inf,
    test_debye_waller_is_unity_at_zero_and_monotone_down,
    test_width_correction_boosts_high_frequency_magnitude,
    test_short_sample_name_handles_si_and_cnt,
    test_samples_from_dataset_extracts_records,
    test_samples_from_dataset_raises_when_no_inverted_samples,
]


def main() -> int:
    passed = 0
    failed = 0
    for test in _TESTS:
        try:
            test()
            passed += 1
            print(f"PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(_TESTS)} total)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
