"""Tests for recipe recording, session bundle save/load, and headless replay."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import pipeline_registry as registry
from dataset_core.adapters import session_bundle

_TRANSMISSION_DATA = r"C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans"


# ── Unit tests: the recording wrapper (no data, no heavy deps) ─────────────────


class _FakeDataset:
    pass


class TestRecordingWrapper(unittest.TestCase):
    def test_records_only_explicit_kwargs(self):
        def stage(dataset, half_width_ps=3.0, taper=None):
            dataset.ran = True
            return dataset

        wrapped = registry._make_recording_wrapper("stage", stage, dataset_position=0)
        ds = _FakeDataset()
        wrapped(ds, half_width_ps=5.0)

        self.assertEqual(len(ds.recipe), 1)
        step = ds.recipe[0]
        self.assertEqual(step["stage"], "stage")
        self.assertEqual(step["kwargs"], {"half_width_ps": 5.0})  # taper default NOT recorded
        self.assertIn("ts", step)
        self.assertTrue(ds.ran)

    def test_config_and_dataset_excluded(self):
        def stage(dataset, config=None, thickness_m=1e-3):
            return dataset

        wrapped = registry._make_recording_wrapper("stage", stage, dataset_position=0)
        ds = _FakeDataset()
        wrapped(ds, config={"big": "dict"}, thickness_m=5e-4)
        self.assertEqual(ds.recipe[0]["kwargs"], {"thickness_m": 5e-4})

    def test_reentrancy_records_only_outer(self):
        ds = _FakeDataset()

        def inner(dataset):
            return dataset

        wrapped_inner = registry._make_recording_wrapper("inner", inner, 0)

        def outer(dataset):
            wrapped_inner(dataset)  # nested call must NOT be recorded
            return dataset

        wrapped_outer = registry._make_recording_wrapper("outer", outer, 0)
        wrapped_outer(ds)

        self.assertEqual([s["stage"] for s in ds.recipe], ["outer"])

    def test_recording_can_be_disabled(self):
        def stage(dataset):
            return dataset

        wrapped = registry._make_recording_wrapper("stage", stage, 0)
        ds = _FakeDataset()
        ds._recipe_recording = False
        wrapped(ds)
        self.assertFalse(getattr(ds, "recipe", []))


# ── Workflow test: full transmission run -> save -> load -> replay ─────────────


@unittest.skipUnless(os.path.isdir(_TRANSMISSION_DATA), "silicon transmission data not present")
class TestSessionWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataset_core import DataSet
        from dataset_core.adapters import thz_adapter as thz

        registry.activate_recording()

        cls.config = {
            "general": {"show_graph": False},
            "resolution": {"limit_to_instrument_resolution": False},
            "sample": {"thickness_m": 500e-6},
            "regions": {"pulse": None},
            "window": {"type": "tukey", "alpha": 0.1, "half_width_ps": 5.0,
                       "interactive_peak": False, "peak_ps": None, "snap_halfwidth_ps": 0.1},
            "pad": {"extend_factor": 1.0},
            "transfer": {"self_reference": False, "apply_snr_mask": True,
                         "min_ref_amp_rel": 1e-3, "regularization_eps": 1e-30, "unwrap_phase": True},
            "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
            "derive": {"eps_background": 11.7},
            "phase_offset": {"band_thz": (0.3, 2.0), "use_snr_mask": True},
        }

        ds = DataSet(_TRANSMISSION_DATA, config=cls.config)
        ds.load_all_data(case_insensitive=True, explicit_dir=True)
        ds.group_files(keywords=["type"])
        thz.subtract_baseline(ds)
        thz.window_time_fixed_width(ds, half_width_ps=5.0, region_ps=None, peak_ps=None,
                                    interactive=False, snap_halfwidth_ps=0.1, center_in_trace=True,
                                    show_graph=False)
        thz.zero_pad(ds, show_graph=False)
        thz.fft_spectrum(ds)
        thz.transfer_function(ds, ref_type="reference")
        thz.remove_phase_offset(ds)
        thz.invert_nk(ds, thickness_m=500e-6)
        thz.derive_eps_sigma(ds)
        cls.dataset = ds

    def _ntype_sigma(self, dataset):
        for fn, obj in dataset.data.data_dict.items():
            if "n-type" in fn.lower():
                return np.asarray(obj.processing_dict["sigma"])
        raise AssertionError("n-type sample not found")

    def test_recipe_captured(self):
        stages = [s["stage"] for s in self.dataset.recipe]
        # setup + processing calls recorded in order
        self.assertIn("load_all_data", stages)
        self.assertIn("group_files", stages)
        self.assertIn("transfer_function", stages)
        self.assertIn("invert_nk", stages)
        self.assertIn("derive_eps_sigma", stages)
        self.assertEqual(stages.index("transfer_function") < stages.index("invert_nk"), True)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = os.path.join(tmp, "run.thzbundle")
            session_bundle.save_session(self.dataset, bundle, notes="unit test")
            self.assertTrue(os.path.exists(os.path.join(bundle, "recipe.json")))
            self.assertTrue(os.path.exists(os.path.join(bundle, "snapshot.pkl")))
            self.assertTrue(os.path.exists(os.path.join(bundle, "report.md")))

            reloaded = session_bundle.load_session(bundle)
            # config + recipe restored
            self.assertEqual(reloaded.config["sample"]["thickness_m"], 500e-6)
            self.assertTrue(len(reloaded.recipe) > 0)
            # processing results restored WITHOUT recompute
            np.testing.assert_allclose(
                self._ntype_sigma(reloaded), self._ntype_sigma(self.dataset), equal_nan=True,
            )

    def test_replay_reproduces_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = os.path.join(tmp, "run.thzbundle")
            session_bundle.save_session(self.dataset, bundle)
            recipe = session_bundle.read_recipe(bundle)
            replayed = registry.replay_recipe(recipe, verbose=False)
            np.testing.assert_allclose(
                self._ntype_sigma(replayed), self._ntype_sigma(self.dataset),
                rtol=1e-6, atol=1e-6, equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
