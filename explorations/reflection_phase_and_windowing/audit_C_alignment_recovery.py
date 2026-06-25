"""Audit: does the self-reference C = Y1_r/Y1_s diagnose (and correct) misalignment?

Physical claim under test (Samuel, 2026-06-24): the SiO2 window faces are not flat,
so the front (Y1) and back (Y2) reflections separate at the parabolic focus. The back
reflection (the sample) is meticulously aligned to the gate focus, so its detection
response D2 is matched between sample and reference. The FRONT spot lands differently
when the window is rotated (CNT-0deg vs CNT-90deg), so D1 differs between acquisitions.

Math consequence:  H_selfref = (Y2s/Y1s)/(Y2r/Y1r) = (Y2s/Y2r) * (Y1r/Y1s) = (Y2s/Y2r) * C.
So  H_selfref / C  ==  Y2s/Y2r  (the direct back-reflection ratio). If D2 is matched and
only the front spot drifts, Y2s/Y2r is the clean estimator and C is exactly the corruption
the front-pulse referencing injects. This script checks that against the real data.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = r'C:\Users\Samuel\matchbook\thz'
sys.path.insert(0, REPO)
from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

DATA = r'C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\main_alignment_tests'
OUT = os.path.join(REPO, 'explorations', 'reflection_phase_and_windowing')

config = {
    "general": {"show_graph": False},
    "geometry": {"theta_external_deg": 45.0, "polarization": "s", "n_sio2": 1.95},
    "regions": {"first_reflection": (152.5, 159), "second_reflection": (177, 183.8)},
    "centering": {"mode": "crop"},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 3.0},
    "pad": {"extend_factor": 1.0},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "transfer": {"self_reference": True, "apply_snr_mask": True,
                 "min_ref_amp_rel": 1e-3, "regularization_eps": 1e-30, "unwrap_phase": True},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "derive": {"eps_background": 11.7},
}

dataset = DataSet(DATA, config=config)
dataset.load_all_data()
print("loaded:", sorted(dataset.data.data_dict.keys()))
thz.build_full_trace_reflection(dataset)
dataset.group_files(keywords=['type'])
thz.subtract_baseline(dataset)
thz.define_reflection_regions(dataset, config)
thz.window_pulses_fixed_width(dataset, half_width_ps=3.0, show_graph=False)
n_fft = config['fft']['n_fft']
thz.fft_spectrum(dataset, segment='second_reflection', n_fft=n_fft)
thz.fft_spectrum(dataset, segment='first_reflection', n_fft=n_fft)
thz.transfer_function(dataset, ref_type='reference')

# collect decomposition per sample
fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
fig.suptitle('Self-reference decomposition vs alignment (main_alignment_tests)')
summary = []
for fname, obj in dataset.data.items():
    if dataset.data.is_reference(fname):
        continue
    p = obj.processing_dict
    diag = p.get('transfer_metrics', {}).get('diagnostics', {})
    f = p.get('fft_freq')
    H = p.get('transfer_H')
    C = p.get('selfref_correction')          # Y1r/Y1s
    Ws = diag.get('W_samp'); Wr = diag.get('W_ref')
    if H is None or C is None:
        continue
    fthz = f * 1e-12
    band = (fthz >= 0.2) & (fthz <= 2.5)
    H_corr = H / C                            # == Y2s/Y2r
    finiteC = np.isfinite(C) & band
    Cmag = np.abs(C)[finiteC]
    summary.append((fname, np.nanmedian(Cmag), np.nanstd(Cmag),
                    np.nanmedian(np.abs(np.angle(C)[finiteC]))))
    axes[0,0].plot(fthz[band], np.abs(C)[band], label=fname)
    axes[0,1].plot(fthz[band], np.angle(C)[band], label=fname)
    axes[1,0].plot(fthz[band], np.abs(H)[band], label=f'H {fname}')
    axes[1,0].plot(fthz[band], np.abs(H_corr)[band], '--', label=f'H/C=Y2s/Y2r {fname}')
    axes[1,1].plot(fthz[band], np.unwrap(np.angle(H))[band], label=f'H {fname}')
    axes[1,1].plot(fthz[band], np.unwrap(np.angle(H_corr))[band], '--', label=f'H/C {fname}')

axes[0,0].set_title('|C| = |Y1r/Y1s|  (1.0 = front spots matched)'); axes[0,0].axhline(1.0, color='k', lw=0.6, ls=':')
axes[0,1].set_title('arg C')
axes[1,0].set_title('|H| (solid) vs |H/C|=|Y2s/Y2r| (dashed)')
axes[1,1].set_title('phase H vs H/C')
for ax in axes.flat:
    ax.set_xlabel('Frequency (THz)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.savefig(os.path.join(OUT, 'audit_C_alignment_recovery.png'), dpi=110)
print("\n=== C = Y1_r/Y1_s summary (0.2-2.5 THz band) ===")
print(f"{'sample':45s} {'med|C|':>8s} {'std|C|':>8s} {'med|argC|(rad)':>14s}")
for name, med, std, ph in summary:
    print(f"{name:45s} {med:8.3f} {std:8.3f} {ph:14.3f}")
print("\nsaved audit_C_alignment_recovery.png")
