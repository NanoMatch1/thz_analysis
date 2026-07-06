"""Tests for headless Drude fitting + KK-consistent geometry optimisation."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thz_core.thz_core.fitting import drude_conductivity
from dataset_core.adapters import conductivity_fitting as cfit

_EPS0 = 8.854187817e-12


# ── Minimal fake dataset (matches the thz._sample_items / _select_sample API) ──


class _FakeSample:
    def __init__(self, freq, sigma, mask):
        self.processing_dict = {'fft_freq': freq, 'sigma': sigma, 'transfer_mask': mask}


class _FakeData(dict):
    def is_reference(self, name):
        return name.startswith('reference')


class _FakeDataset:
    def __init__(self, samples: dict):
        self.data = _FakeData(samples)
        self.config = {}


def _make_freq_mask(n=200, f_max_thz=2.5):
    freq = np.linspace(0.0, f_max_thz, n) / cfit._HZ_TO_THZ
    mask = (freq > 0.3 / cfit._HZ_TO_THZ) & (freq < 2.0 / cfit._HZ_TO_THZ)
    return freq, mask


class TestJointDrudeFit(unittest.TestCase):
    def test_recovers_planted_drude(self):
        freq, mask = _make_freq_mask()
        omega = 2 * np.pi * freq
        sigma_dc_true, tau_true = 320.0, 1.8e-13
        sigma = drude_conductivity(omega, sigma_dc_true, tau_true)
        result, _ = cfit.joint_drude_fit(
            freq, sigma, mask, fit_band_thz=(0.3, 2.0),
            initial={'sigma_dc': 100.0, 'tau': 1e-13},
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.param_values['sigma_dc'], sigma_dc_true, delta=1.0)
        self.assertAlmostEqual(result.param_values['tau'], tau_true, delta=2e-15)
        self.assertGreater(result.r_squared, 0.999)

    def test_offset_sigma2_lowers_joint_r2(self):
        """A sigma_2-only offset (the thickness/eps_inf artefact) must hurt the JOINT fit."""
        freq, mask = _make_freq_mask()
        omega = 2 * np.pi * freq
        clean = drude_conductivity(omega, 320.0, 1.8e-13)
        # add an omega-proportional imaginary offset == a wrong-eps_inf / wrong-d signature
        corrupted = clean + 1j * omega * _EPS0 * 1.0
        r2_clean = cfit.joint_drude_fit(freq, clean, mask, fit_band_thz=(0.3, 2.0))[0].r_squared
        r2_bad = cfit.joint_drude_fit(freq, corrupted, mask, fit_band_thz=(0.3, 2.0))[0].r_squared
        self.assertGreater(r2_clean, 0.999)
        self.assertLess(r2_bad, r2_clean)


class TestOptimizeGeometry(unittest.TestCase):
    def _build_dataset(self):
        freq, mask = _make_freq_mask()
        omega = 2 * np.pi * freq
        self.freq, self.mask, self.omega = freq, mask, omega
        self.true_value = 500e-6
        # two samples with different Drude params, same "true" geometry
        drude_a = drude_conductivity(omega, 320.0, 1.8e-13)
        drude_b = drude_conductivity(omega, 90.0, 2.4e-13)
        self._drude = {'sample_a': drude_a, 'reference': None, 'sample_b': drude_b}
        samples = {
            'sample_a': _FakeSample(freq, drude_a.copy(), mask),
            'reference': _FakeSample(freq, None, mask),
            'sample_b': _FakeSample(freq, drude_b.copy(), mask),
        }
        return _FakeDataset(samples)

    def _apply_value(self, dataset, value):
        """Synthetic geometry: sigma = true Drude + sigma_2 offset ∝ (value - true).

        Mirrors the real mechanism — a wrong thickness adds an omega-proportional
        imaginary term (via n^2), which only the correct value removes.
        """
        offset = (value - self.true_value) / 1e-6 * 0.4  # eps-units per micrometre
        for name, obj in dataset.data.items():
            if dataset.data.is_reference(name):
                continue
            clean = self._drude[name]
            obj.processing_dict['sigma'] = clean + 1j * self.omega * _EPS0 * offset

    def test_finds_kk_consistent_value(self):
        dataset = self._build_dataset()
        out = cfit.optimize_geometry_parameter(
            dataset, self._apply_value, bounds=(460e-6, 540e-6),
            param_label="thickness", display_scale=1e6, display_unit="um",
            fit_band_thz=(0.3, 2.0), n_scan=21, verbose=False,
        )
        self.assertAlmostEqual(out['best_value'], self.true_value, delta=2e-6)
        self.assertGreater(out['best_joint_r2'], 0.99)
        # dataset left at the optimum
        for name, obj in dataset.data.items():
            if dataset.data.is_reference(name):
                continue
            self.assertTrue(np.allclose(obj.processing_dict['sigma'], self._drude[name], atol=1e-6))

    def test_scan_curve_is_convex_about_optimum(self):
        dataset = self._build_dataset()
        out = cfit.optimize_geometry_parameter(
            dataset, self._apply_value, bounds=(460e-6, 540e-6),
            fit_band_thz=(0.3, 2.0), n_scan=21, refine=False, verbose=False,
        )
        badness = out['scan_badness']
        argmin = int(np.argmin(badness))
        self.assertGreater(argmin, 0)
        self.assertLess(argmin, len(badness) - 1)


class TestSliderHeadless(unittest.TestCase):
    def test_slider_builds_without_blocking(self):
        freq, mask = _make_freq_mask()
        omega = 2 * np.pi * freq
        sigma = drude_conductivity(omega, 320.0, 1.8e-13)
        samples = {
            'sample_a': _FakeSample(freq, sigma.copy(), mask),
            'reference': _FakeSample(freq, None, mask),
        }
        dataset = _FakeDataset(samples)

        def apply_value(ds, value):
            for name, obj in ds.data.items():
                if ds.data.is_reference(name):
                    continue
                obj.processing_dict['sigma'] = sigma.copy()  # geometry-independent toy

        handles = cfit.thickness_slider(
            dataset, (460e-6, 540e-6), reinvert=apply_value,
            init_value_m=500e-6, fit_band_thz=(0.3, 2.0), show=False,
        )
        self.assertIn('figure', handles)
        self.assertIn('slider', handles)
        # GC guard: the widgets must be pinned to the figure so callbacks survive the
        # caller discarding the return value (the cause of the frozen-slider bug).
        self.assertTrue(hasattr(handles['figure'], '_thickness_slider_widgets'))
        self.assertIn(handles['slider'], handles['figure']._thickness_slider_widgets)
        # driving the slider redraw directly must not raise
        handles['redraw'](495e-6)
        # driving through the ACTUAL widget callback path (on_changed) must also work
        handles['slider'].set_val(510.0)  # micrometres


if __name__ == '__main__':
    unittest.main(verbosity=2)
