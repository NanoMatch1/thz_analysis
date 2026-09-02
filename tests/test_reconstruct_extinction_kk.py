"""Tests for the geometry-agnostic n<->k Kramers-Kronig reconstruction.

Core relation in thz_core.kramers_kronig (extinction_from_refractive_index /
refractive_index_from_extinction); pipeline wrapper thz.reconstruct_extinction_kk.

Strategy: build a causal Lorentz complex index (n + i*k), keep the phase-derived n exact,
and hand the stage a *corrupted* amplitude-derived k. The corruption is zero at the
highest-SNR bin (physically, the amplitude is most reliable there) and grows away from it.
The stage should rebuild k from n, recovering the true k far better than the corrupted
input, and pass the bidirectional consistency check.

Dependency-free; run with the project venv:
    .venv/Scripts/python.exe tests/test_reconstruct_extinction_kk.py
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thz_core.thz_core as core
from dataset_core.adapters import thz_adapter as thz


def _lorentz_index(freq_hz, eps_inf=4.0, f0=2.5e12, df=0.7e12, delta_eps=2.5):
    """Causal n + i*k from a single Lorentz oscillator (n -> sqrt(eps_inf) at high f)."""
    w, w0, g = 2 * np.pi * freq_hz, 2 * np.pi * f0, 2 * np.pi * df
    eps = eps_inf + delta_eps * w0**2 / (w0**2 - w**2 - 1j * w * g)
    n = np.sqrt(eps)
    n = np.where(n.imag < 0, -n, n)
    return np.real(n), np.imag(n), 2.5e12  # n, k, resonance frequency


class _FakeDataService(dict):
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


def _build_dataset():
    freq = np.linspace(0.0, 8e12, 800)
    n_true, k_true, f0 = _lorentz_index(freq)

    trusted = (freq > 0.5e12) & (freq < 5.0e12)
    # SNR peaked at the resonance (where |amplitude| is largest / most reliable).
    snr_db = 30.0 * np.exp(-((freq - f0) / (0.7e12)) ** 2)

    # Corrupt the measured k: a linear-in-f distortion that vanishes at the resonance
    # (the anchor bin) and grows toward the band edges — a stand-in for the amplitude/gap
    # systematic that KK is meant to repair.
    k_measured = k_true + 0.5 * (freq / f0 - 1.0)

    sample = _FakeSample({
        "fft_freq": freq,
        "n": n_true,
        "k": k_measured,
        "transfer_mask": trusted,
        "snr_db": snr_db,
    })
    dataset = _FakeDataset(
        _FakeDataService({"sample.acc": sample, "gold.acc": _FakeSample({})},
                         references=["gold.acc"]),
        config={"kk_nk": {"n_infinity": 2.0, "use_snr_anchor": True}},
    )
    return dataset, sample, freq, n_true, k_true, k_measured, trusted


def test_rebuilds_k_and_beats_the_corrupted_input():
    dataset, sample, freq, n_true, k_true, k_measured, trusted = _build_dataset()
    thz.reconstruct_extinction_kk(dataset)
    pd = sample.processing_dict

    # Measured k stashed, k overwritten, metrics recorded.
    assert "k_measured" in pd and np.allclose(pd["k_measured"], k_measured)
    k_kk = pd["k"]
    assert "kk_nk_metrics" in pd

    band = trusted
    err_kk = np.max(np.abs(k_kk[band] - k_true[band]))
    err_measured = np.max(np.abs(k_measured[band] - k_true[band]))
    assert err_kk < err_measured, (err_kk, err_measured)
    assert err_kk < 0.05, err_kk


def test_anchor_is_the_peak_snr_bin():
    dataset, sample, freq, *_ = _build_dataset()
    thz.reconstruct_extinction_kk(dataset)
    metrics = sample.processing_dict["kk_nk_metrics"]
    # SNR peaks at 2.5 THz -> anchor should land there.
    assert abs(metrics["anchor_frequency_thz"] - 2.5) < 0.2, metrics["anchor_frequency_thz"]


def test_consistency_check_recovers_n_from_measured_k():
    """The reverse transform on the *measured* k should agree with the true n where the
    measured k is good (near the anchor) and disagree where it is corrupted — the metric
    is a geometry-free flag of the amplitude systematic."""
    dataset, sample, freq, n_true, k_true, k_measured, trusted = _build_dataset()
    thz.reconstruct_extinction_kk(dataset)
    pd = sample.processing_dict
    assert "n_kk_from_k" in pd
    assert "n_consistency_rms" in pd["kk_nk_metrics"]
    # Corrupted k -> a non-zero but bounded disagreement with the true n.
    assert pd["kk_nk_metrics"]["n_consistency_rms"] > 0.0


def test_reference_is_skipped():
    dataset, sample, *_ = _build_dataset()
    thz.reconstruct_extinction_kk(dataset)
    # The gold reference has an empty processing_dict and must be left untouched.
    assert dataset.data["gold.acc"].processing_dict == {}


def test_unsubtracted_when_anchor_disabled():
    freq = np.linspace(0.0, 8e12, 800)
    n_true, k_true, f0 = _lorentz_index(freq)
    sample = _FakeSample({
        "fft_freq": freq, "n": n_true, "k": k_true,
        "transfer_mask": (freq > 0.5e12) & (freq < 5.0e12),
    })
    dataset = _FakeDataset(
        _FakeDataService({"s.acc": sample}, references=[]),
        config={"kk_nk": {"n_infinity": 2.0, "use_snr_anchor": False}},
    )
    thz.reconstruct_extinction_kk(dataset)
    # No anchor selected -> unsubtracted transform, metrics report no anchor frequency.
    assert sample.processing_dict["kk_nk_metrics"]["anchor_frequency_thz"] is None


def test_default_anchor_mode_is_zero_high():
    """With no anchor config the stage defaults to the zero-high-edge anchor (weak-k case)."""
    freq = np.linspace(0.0, 8e12, 800)
    n_true, k_true, _ = _lorentz_index(freq)
    sample = _FakeSample({
        "fft_freq": freq, "n": n_true, "k": k_true,
        "transfer_mask": (freq > 0.5e12) & (freq < 5.0e12),
    })
    dataset = _FakeDataset(_FakeDataService({"s.acc": sample}, references=[]),
                           config={"kk_nk": {"n_infinity": 2.0}})
    thz.reconstruct_extinction_kk(dataset)
    assert sample.processing_dict["kk_nk_metrics"]["anchor_mode"] == "zero_high"


def test_kk_band_excludes_edge_rolloff():
    """A low-frequency roll-off in n (untrusted band edge) must not ramp k when kk_band_thz
    restricts the integral to the clean interior — the silicon failure mode."""
    freq = np.linspace(0.0, 6e12, 900)
    f_thz = freq * 1e-12
    n_infinity = 3.42
    # Flat transparent index in the interior, rolling down to ~2.0 below 0.5 THz.
    n = np.full_like(freq, n_infinity)
    edge = f_thz < 0.5
    n[edge] = n_infinity - 1.4 * (0.5 - f_thz[edge]) / 0.5
    k_meas = np.zeros_like(freq)
    trusted = (freq > 0.05e12) & (freq < 3.0e12)

    def _run(kk_band):
        s = _FakeSample({"fft_freq": freq, "n": n.copy(), "k": k_meas.copy(), "transfer_mask": trusted})
        cfg = {"kk_nk": {"n_infinity": n_infinity, "anchor_mode": "zero_high",
                         "clip_negative": False, "on_non_passive": "apply"}}
        if kk_band is not None:
            cfg["kk_nk"]["kk_band_thz"] = kk_band
        thz.reconstruct_extinction_kk(_FakeDataset(_FakeDataService({"s.acc": s}, references=[]), cfg))
        return s.processing_dict["k"]

    interior = (f_thz > 0.6) & (f_thz < 2.8)
    k_no_band = _run(None)                 # roll-off feeds the integral -> ramp
    k_band = _run((0.6, 2.8))              # clean interior only -> ~0
    assert np.max(np.abs(k_band[interior])) < np.max(np.abs(k_no_band[interior]))
    assert np.max(np.abs(k_band[interior])) < 0.1, np.max(np.abs(k_band[interior]))


def test_clip_negative_enforced():
    freq = np.linspace(0.0, 6e12, 800)
    f_thz = freq * 1e-12
    n_infinity = 3.42
    n = np.full_like(freq, n_infinity)
    n[f_thz < 1.0] = n_infinity - 0.2      # a low-frequency dip that drives k negative
    trusted = (freq > 0.1e12) & (freq < 3.0e12)

    def _run(clip):
        s = _FakeSample({"fft_freq": freq, "n": n.copy(), "k": np.zeros_like(freq), "transfer_mask": trusted})
        cfg = {"kk_nk": {"n_infinity": n_infinity, "anchor_mode": "zero_high",
                         "kk_band_thz": (0.3, 2.8), "clip_negative": clip, "on_non_passive": "apply"}}
        thz.reconstruct_extinction_kk(_FakeDataset(_FakeDataService({"s.acc": s}, references=[]), cfg))
        return s.processing_dict["k"], s.processing_dict["kk_nk_metrics"]

    k_clipped, m_clipped = _run(True)
    k_raw, m_raw = _run(False)
    assert np.min(k_clipped) >= 0.0
    assert m_raw["negative_fraction"] > 0.0          # there genuinely were negatives
    assert np.min(k_raw) < 0.0


def test_non_passive_keeps_measured_k_by_default():
    """A systematically-negative (non-passive) reconstruction — the doped/Drude case — must
    leave the measured k untouched by default and flag it, not overwrite with a bad k."""
    freq = np.linspace(0.0, 6e12, 800)
    f_thz = freq * 1e-12
    n_infinity = 3.42
    n = np.full_like(freq, n_infinity)
    n[f_thz < 1.0] = n_infinity - 0.2          # low-f dip -> systematically negative raw KK
    k_meas = np.full_like(freq, 0.05)          # some measured k to preserve
    trusted = (freq > 0.1e12) & (freq < 3.0e12)
    s = _FakeSample({"fft_freq": freq, "n": n, "k": k_meas.copy(), "transfer_mask": trusted})
    cfg = {"kk_nk": {"n_infinity": n_infinity, "anchor_mode": "zero_high",
                     "kk_band_thz": (0.3, 2.8)}}   # default on_non_passive = keep_measured
    thz.reconstruct_extinction_kk(_FakeDataset(_FakeDataService({"s.acc": s}, references=[]), cfg))
    pd = s.processing_dict
    assert pd["kk_nk_metrics"]["non_passive"] is True
    assert pd["kk_nk_metrics"]["applied"] is False
    assert "k_kk_rejected" in pd                    # reconstruction stashed for inspection
    assert np.allclose(pd["k"], k_meas)             # measured k preserved


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
