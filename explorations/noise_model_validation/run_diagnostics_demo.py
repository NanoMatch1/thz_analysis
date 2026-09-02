"""Generate the assumptions ledger, then run the diagnostics against a real dataset.

Shows both halves of the single-source-of-truth arrangement: the same ``@diagnostic``
registrations produce the document AND the runtime findings.

    python explorations/noise_model_validation/run_diagnostics_demo.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from dataset_core import DataSet
from dataset_core.adapters import diagnostics
from dataset_core.adapters import thz_adapter as thz

DATA_DIR = "/home/match/data/CNTs/2026-08-20_CNT-paper-doped_2/export"
LEDGER = os.path.join(REPO_ROOT, "docs", "assumptions_ledger.md")

config = {
    "general": {"show_graph": False},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 5},
    "baseline": {"n_points": 10},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "resolution": {"limit_to_instrument_resolution": False, "broadening_factor": 1.0},
    "transfer": {"self_reference": False, "apply_snr_mask": True,
                 "min_ref_amp_rel": 1e-3, "regularization_eps": 1e-30,
                 "unwrap_phase": True},
}

print("=" * 78)
print("THE LEDGER (generated from the registry, never hand-edited)")
print("=" * 78)
diagnostics.write_ledger(LEDGER)
print(f"    {len(diagnostics.DIAGNOSTIC_REGISTRY)} assumptions across stages: "
      f"{', '.join(diagnostics.registered_stages())}")

print()
print("=" * 78)
print("A REAL RUN")
print("=" * 78)
dataset = DataSet(DATA_DIR, config=config)
dataset.load_all_data()
thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
dataset.group_files(keywords=["type"])
thz.subtract_baseline(dataset)
thz.window_single_pulse_fixed_width(dataset, half_width_ps=5, show_graph=False)
thz.fft_spectrum(dataset, n_fft=config["fft"]["n_fft"])
thz.transfer_function(dataset, ref_type="reference")
thz.compute_instrument_resolution(dataset, config)

diagnostics.run_report(dataset, config)
