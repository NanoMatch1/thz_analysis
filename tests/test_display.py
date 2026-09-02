"""Tests for the reusable registry-driven display layer (``dataset_core.adapters.display``).

Two forms, per the project's testing philosophy:
  * discrete unit tests — item selection, style resolution priority, data access, styling helper
  * a headless workflow test — every misalignment figure builds on an Agg backend without raising
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import display


# ── synthetic dataset matching the interface display consumes ─────────────────


def _processing_dict(n_value: float = 3.4) -> dict:
    freq = np.linspace(0.0, 2.5e12, 120)
    n = np.full_like(freq, n_value)
    k = np.linspace(0.4, 0.02, freq.size)
    eps = (n + 1j * k) ** 2
    sigma = 200.0 / (1 + 1j * freq / 1e12)
    fft_spectrum = np.exp(-((freq - 0.6e12) / 3e11) ** 2) * np.exp(1j * freq / 1e12)
    transfer_H = 0.5 * np.exp(-1j * 0.3 * freq / 1e12)
    mask = (freq > 0.3e12) & (freq < 2.0e12)
    time_s = np.linspace(0, 50e-12, 300)
    amplitude = np.exp(-((time_s - 25e-12) / 2e-12) ** 2) * np.sin(time_s * 1e12)
    return {
        "fft_freq": freq, "fft_spectrum": fft_spectrum,
        "transfer_H": transfer_H, "transfer_mask": mask, "snr_mask": mask,
        "n": n, "k": k, "eps": eps, "sigma": sigma,
        "n_sigma": np.full_like(freq, 0.05), "k_sigma": np.full_like(freq, 0.02),
        "time_domain": np.column_stack((time_s, amplitude)),
    }


class _FakeObj:
    def __init__(self, pd):
        self.processing_dict = pd


class _FakeData:
    """Mimics DataService: a ``data_dict`` mapping + reference classification by name."""

    def __init__(self, mapping):
        self.data_dict = dict(mapping)

    def is_reference(self, name):
        return name.startswith("reference")


class _FakeDataset:
    def __init__(self, mapping):
        self.data = _FakeData(mapping)
        self.config = {"resolution": {"limit_to_instrument_resolution": True}}


def _misalignment_dataset():
    return _FakeDataset({
        "sample_gold_minus-4-mrad.acc": _FakeObj(_processing_dict(3.2)),
        "sample_gold_minus-2-mrad.acc": _FakeObj(_processing_dict(3.3)),
        "reference_gold_plus-0-mrad.acc": _FakeObj(_processing_dict(3.4)),
        "sample_gold_plus-2-mrad.acc": _FakeObj(_processing_dict(3.5)),
        "sample_gold_plus-4-mrad.acc": _FakeObj(_processing_dict(3.6)),
    })


# ── item selection ────────────────────────────────────────────────────────────


class TestItemSelection(unittest.TestCase):
    def setUp(self):
        self.dataset = _misalignment_dataset()

    def test_none_excludes_references_by_default(self):
        selected = [f for f, _ in display._iter_selected_items(
            self.dataset, files=None, include_references=False)]
        self.assertNotIn("reference_gold_plus-0-mrad.acc", selected)
        self.assertEqual(len(selected), 4)

    def test_none_includes_references_when_flagged(self):
        selected = [f for f, _ in display._iter_selected_items(
            self.dataset, files=None, include_references=True)]
        self.assertIn("reference_gold_plus-0-mrad.acc", selected)
        self.assertEqual(len(selected), 5)

    def test_regex_string_selector(self):
        selected = [f for f, _ in display._iter_selected_items(
            self.dataset, files="minus", include_references=True)]
        self.assertEqual(len(selected), 2)

    def test_callable_selector(self):
        selected = [f for f, _ in display._iter_selected_items(
            self.dataset, files=lambda name: "plus-4" in name, include_references=True)]
        self.assertEqual(selected, ["sample_gold_plus-4-mrad.acc"])


# ── style resolution priority ───────────────────────────────────────────────────


class TestStyleResolution(unittest.TestCase):
    def test_priority_order(self):
        style = {
            "default": {"alpha": 0.5, "color": "green"},
            "style_fn": lambda f, o: {"color": "blue", "zorder": 4},
            "per_file": {"gold": {"color": "purple"},          # regex/substring
                         "sample_gold_plus-2-mrad.acc": {"color": "red"}},  # exact
        }
        cycle = ["C0", "C1"]
        # exact match beats regex beats style_fn beats default
        exact = display.resolve_item_style(
            "sample_gold_plus-2-mrad.acc", None, style, 0, cycle, {"marker": "o"})
        self.assertEqual(exact["color"], "red")
        self.assertEqual(exact["zorder"], 4)      # from style_fn, not overridden
        self.assertEqual(exact["alpha"], 0.5)     # from default, not overridden
        # a file matching only the regex layer
        regex_only = display.resolve_item_style(
            "sample_gold_minus-2-mrad.acc", None, style, 1, cycle, {"marker": "o"})
        self.assertEqual(regex_only["color"], "purple")

    def test_color_cycle_fallback(self):
        cycle = ["A", "B", "C"]
        # unset color falls back to color_cycle[index % len]
        self.assertEqual(
            display.resolve_item_style("x", None, None, 1, cycle, {})["color"], "B")
        self.assertEqual(
            display.resolve_item_style("x", None, None, 3, cycle, {})["color"], "A")

    def test_label_defaults_to_filename(self):
        resolved = display.resolve_item_style("foo.acc", None, None, 0, ["A"], {})
        self.assertEqual(resolved["label"], "foo.acc")


# ── data access ──────────────────────────────────────────────────────────────


class TestGetSeries(unittest.TestCase):
    def setUp(self):
        self.dataset = _misalignment_dataset()

    def test_get_series_returns_aligned_arrays(self):
        series = display.get_series(self.dataset, "n")
        self.assertNotIn("reference_gold_plus-0-mrad.acc", series)  # n excludes refs
        one = next(iter(series.values()))
        self.assertEqual(one["freq_thz"].shape, one["y"].shape)
        self.assertEqual(one["error"].shape, one["y"].shape)   # n_sigma aligned
        self.assertTrue(np.allclose(one["freq_thz"], one["freq_hz"] * 1e-12))

    def test_fft_series_includes_references(self):
        series = display.get_series(self.dataset, "fft_mag")
        self.assertIn("reference_gold_plus-0-mrad.acc", series)

    def test_unknown_quantity_raises(self):
        with self.assertRaises(KeyError):
            display.get_series(self.dataset, "not_a_quantity")

    def test_available_quantities(self):
        names = display.available_quantities(self.dataset)
        for expected in ["fft_mag", "fft_phase", "n", "k", "transfer_mag"]:
            self.assertIn(expected, names)


# ── styling helper (diverging series) ────────────────────────────────────────


class TestDivergingStyle(unittest.TestCase):
    def setUp(self):
        self.dataset = _misalignment_dataset()

    def test_parse_signed_displacement(self):
        self.assertEqual(
            display.parse_signed_displacement("x_gold_minus-4-mrad.acc", "_gold_"), -4.0)
        self.assertEqual(
            display.parse_signed_displacement("x_gold_plus-2-mrad.acc", "_gold_"), 2.0)
        self.assertEqual(
            display.parse_signed_displacement("x_gold_plus-0-mrad.acc", "_gold_"), 0.0)
        self.assertIsNone(
            display.parse_signed_displacement("unrelated.acc", "_gold_"))

    def test_style_fn_emphasises_zero_and_labels(self):
        style = display.diverging_series_style(self.dataset, "_gold_")
        style_fn = style["style_fn"]
        zero = style_fn("reference_gold_plus-0-mrad.acc", None)
        self.assertEqual(zero["color"], "black")
        self.assertEqual(zero["zorder"], 6)
        self.assertIn("+0.0 mrad", zero["label"])
        negative = style_fn("sample_gold_minus-4-mrad.acc", None)
        self.assertNotEqual(negative["color"], "black")
        self.assertEqual(negative["zorder"], 2)
        # a file outside the series gets no styling
        self.assertEqual(style_fn("unrelated.acc", None), {})


# ── headless workflow: every misalignment figure builds ──────────────────────


class TestFiguresBuildHeadless(unittest.TestCase):
    def setUp(self):
        self.dataset = _misalignment_dataset()
        self.style = display.diverging_series_style(self.dataset, "_gold_")

    def tearDown(self):
        plt.close("all")

    def test_plot_quantity_all_freq_panels(self):
        for quantity in ["fft_mag", "fft_phase", "n", "k", "transfer_mag", "transfer_phase"]:
            fig = display.plot_quantity(self.dataset, quantity, style=self.style,
                                        plot_config={"x_range": (0, 2.5)})
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(fig.axes[0].get_title())

    def test_plot_time_domain(self):
        fig = display.plot_time_domain(self.dataset, style=self.style)
        self.assertIsInstance(fig, plt.Figure)
        # all five traces (references included) drawn as lines
        self.assertEqual(len(fig.axes[0].lines), 5)

    def test_errorbars_and_mask_toggle(self):
        # error bars off + no mask must still build
        fig = display.plot_quantity(self.dataset, "n",
                                    plot_config={"show_errorbars": False, "show_mask": False})
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_into_existing_axis(self):
        fig, ax = plt.subplots()
        returned = display.plot_quantity(self.dataset, "n", ax=ax)
        self.assertIs(returned, fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
