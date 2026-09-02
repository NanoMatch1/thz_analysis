"""Phase A: FitGUI error-bar support (data uncertainty drawn as bars).

The GUI is tkinter; these tests construct it (without the blocking mainloop), drive ``_redraw``,
and inspect the axes. They SKIP when no display / Tk is available so headless CI stays green.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thz_core.thz_core.fitting import drude_conductivity


def _make_gui(**kwargs):
    """Construct a FitGUI or skip the test if Tk can't initialise (headless)."""
    import tkinter as tk
    from thz_core.thz_core.fit_gui import FitGUI
    try:
        return FitGUI(**kwargs)
    except tk.TclError as exc:
        raise unittest.SkipTest(f"no display for tkinter: {exc}")


def _synthetic():
    freq = np.linspace(0.1e12, 2.5e12, 150)
    omega = 2 * np.pi * freq
    sigma = drude_conductivity(omega, 320.0, 1.8e-13)
    mask = (freq > 0.3e12) & (freq < 2.0e12)
    error = np.abs(sigma) * 0.05 + 1j * np.abs(sigma) * 0.06
    return freq, sigma, mask, error


class TestFitGuiErrorBars(unittest.TestCase):
    def test_error_bars_drawn_when_provided(self):
        freq, sigma, mask, error = _synthetic()
        gui = _make_gui(freq=freq, data_dict={"a": sigma}, mask_dict={"a": mask},
                        error_dict={"a": error}, initial_model="drude_conductivity")
        try:
            gui._redraw()
            # complex model -> one errorbar container for Re, one for Im
            self.assertEqual(len(gui._ax_main.containers), 2)
        finally:
            gui.root.destroy()

    def test_no_error_bars_without_error(self):
        freq, sigma, mask, _ = _synthetic()
        gui = _make_gui(freq=freq, data_dict={"a": sigma}, mask_dict={"a": mask},
                        initial_model="drude_conductivity")
        try:
            gui._redraw()
            self.assertEqual(len(gui._ax_main.containers), 0)
            self.assertIsNone(gui._error)
        finally:
            gui.root.destroy()

    def test_error_is_per_file(self):
        freq, sigma, mask, error = _synthetic()
        # only file 'a' has an error array; 'b' has none
        gui = _make_gui(freq=freq, data_dict={"a": sigma, "b": sigma},
                        mask_dict={"a": mask, "b": mask}, error_dict={"a": error},
                        initial_model="drude_conductivity")
        try:
            self.assertIsNotNone(gui._error)             # current file is 'a'
            gui._current_file_index = 1                  # switch to 'b'
            self.assertIsNone(gui._error)
        finally:
            gui.root.destroy()


class TestFitCurveFullRange(unittest.TestCase):
    """A completed fit's overlay should extrapolate across the whole frequency axis, not
    just the (narrower) fit_range_hz band -- otherwise users click "Preview" to see the
    extrapolation, which (see TestPreviewDiscardsFit below) silently discards the fit."""

    def test_run_fit_curve_spans_full_frequency_range(self):
        freq, sigma, mask, _ = _synthetic()
        gui = _make_gui(freq=freq, data_dict={"a": sigma}, mask_dict={"a": mask},
                        initial_model="drude_conductivity")
        try:
            # Restrict the fit to a sub-band narrower than the full frequency axis.
            gui._fmin_var.set("0.5")
            gui._fmax_var.set("1.5")
            gui._run_fit()
            self.assertIsNotNone(gui._last_result)
            self.assertTrue(gui._last_result.success)
            self.assertLess(gui._last_result.fit_freq.max(), freq.max())  # fit band is narrower

            gui._redraw()
            fit_lines = [ln for ln in gui._ax_main.get_lines() if ln.get_label().startswith("Fit")]
            self.assertTrue(fit_lines)
            for line in fit_lines:
                xdata = line.get_xdata()
                # The drawn curve covers the FULL axis, not just the fit_range_hz sub-band.
                self.assertAlmostEqual(xdata.min(), freq.min() * 1e-12, places=3)
                self.assertAlmostEqual(xdata.max(), freq.max() * 1e-12, places=3)
        finally:
            gui.root.destroy()


class TestPreviewDiscardsFit(unittest.TestCase):
    """Documents the current behaviour: clicking "Preview (no fit)" after "Run Fit" clears
    _last_result, discarding that fit (it won't be stored / travel to the next file / bundle)
    unless "Run Fit" is clicked again. TestFitCurveFullRange above removes the main reason
    users reached for Preview after a fit (seeing the curve across the full range)."""

    def test_preview_clears_last_result(self):
        freq, sigma, mask, _ = _synthetic()
        gui = _make_gui(freq=freq, data_dict={"a": sigma}, mask_dict={"a": mask},
                        initial_model="drude_conductivity")
        try:
            gui._run_fit()
            self.assertIsNotNone(gui._last_result)
            gui._preview()
            self.assertIsNone(gui._last_result)
        finally:
            gui.root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
