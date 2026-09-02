"""Tests for dataset/session merging and the replay_session editing convenience."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import session_merge, session_bundle, results_viewer

_TRANSMISSION_DATA = r"C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans"


def _build_dataset():
    from dataset_core import DataSet
    from dataset_core.adapters import thz_adapter as thz

    config = {
        "general": {"show_graph": False},
        "resolution": {"limit_to_instrument_resolution": False},
        "sample": {"thickness_m": 500e-6},
        "window": {"type": "tukey", "alpha": 0.1, "half_width_ps": 5.0,
                   "interactive_peak": False, "peak_ps": None, "snap_halfwidth_ps": 0.1},
        "transfer": {"self_reference": False, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                     "regularization_eps": 1e-30, "unwrap_phase": True},
        "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        "derive": {"eps_background": 11.7}, "regions": {"pulse": None}, "pad": {"extend_factor": 1.0},
    }
    ds = DataSet(_TRANSMISSION_DATA, config=config)
    ds.load_all_data(case_insensitive=True, explicit_dir=True)
    ds.group_files(keywords=["type"])
    thz.subtract_baseline(ds)
    thz.window_time_fixed_width(ds, half_width_ps=5.0, region_ps=None, peak_ps=None,
                                interactive=False, snap_halfwidth_ps=0.1, center_in_trace=True,
                                show_graph=False)
    thz.zero_pad(ds, show_graph=False)
    thz.fft_spectrum(ds)
    thz.transfer_function(ds, ref_type="reference")
    thz.invert_nk(ds, thickness_m=500e-6)
    thz.derive_eps_sigma(ds)
    return ds


@unittest.skipUnless(os.path.isdir(_TRANSMISSION_DATA), "silicon transmission data not present")
class TestMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _build_dataset()
        cls.n_files = len(cls.ds.data.data_dict)

    def test_merge_namespaces_and_preserves_classification(self):
        merged = session_merge.merge_datasets([self.ds, self.ds], labels=["run_a", "run_b"])
        # every file appears once per source, namespaced
        self.assertEqual(len(merged.data.data_dict), 2 * self.n_files)
        keys = list(merged.data.data_dict.keys())
        self.assertTrue(all("::" in k for k in keys))
        self.assertTrue(any(k.startswith("run_a::") for k in keys))
        self.assertTrue(any(k.startswith("run_b::") for k in keys))
        # reference classification survives the merge
        for filename in self.ds.data.data_dict:
            was_ref = self.ds.data.is_reference(filename)
            self.assertEqual(merged.data.is_reference(f"run_a::{filename}"), was_ref)
        # provenance recorded
        self.assertEqual(merged.recipe[-1]["stage"], "merge_datasets")
        self.assertEqual(merged.file_metadata[keys[0]]["source"], keys[0].split("::")[0])

    def test_merged_dataset_drives_viewer_and_export(self):
        merged = session_merge.merge_datasets([self.ds, self.ds], labels=["a", "b"])
        viewer = results_viewer.launch_results_viewer(merged, show=False)
        for label in viewer._labels:
            viewer._on_quantity(label)
        with tempfile.TemporaryDirectory() as tmp:
            written = results_viewer.export_quantities(merged, tmp)
            # 2 samples x 2 sources = 4 results CSVs
            self.assertEqual(len([p for p in written if p.endswith("_results.csv")]), 4)

    def test_merge_sessions_from_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            b1 = os.path.join(tmp, "run1.thzbundle")
            b2 = os.path.join(tmp, "run2.thzbundle")
            session_bundle.save_session(self.ds, b1)
            session_bundle.save_session(self.ds, b2)
            merged = session_merge.merge_sessions([b1, b2], labels=["one", "two"])
            self.assertEqual(len(merged.data.data_dict), 2 * self.n_files)
            self.assertIn("sources", merged.recipe[-1]["kwargs"])


@unittest.skipUnless(os.path.isdir(_TRANSMISSION_DATA), "silicon transmission data not present")
class TestReplaySession(unittest.TestCase):
    def test_replay_with_override_and_save(self):
        from dataset_core.adapters import pipeline_registry
        pipeline_registry.activate_recording()
        ds = _build_dataset()

        def ntype_n(dataset):
            for fn, obj in dataset.data.data_dict.items():
                if "n-type" in fn.lower():
                    return np.asarray(obj.processing_dict["n"])
            return None

        with tempfile.TemporaryDirectory() as tmp:
            bundle = os.path.join(tmp, "run.thzbundle")
            session_bundle.save_session(ds, bundle)
            out = os.path.join(tmp, "edited.thzbundle")
            # thickness_m was passed explicitly to invert_nk, so it's a recorded STEP kwarg ->
            # edit it via override_steps (override_config only reaches stages that read config).
            edited = session_bundle.replay_session(
                bundle, override_steps={"invert_nk": {"thickness_m": 400e-6}}, save_to=out)
            self.assertTrue(os.path.exists(os.path.join(out, "recipe.json")))
            n_original = ntype_n(ds)
            n_edited = ntype_n(edited)
            # thinner d -> larger (n-1); the curves must differ
            self.assertFalse(np.allclose(n_original, n_edited, equal_nan=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
