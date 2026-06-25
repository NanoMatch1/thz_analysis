"""Invert H (self-ref) vs H/C (=Y2s/Y2r) and benchmark against physics.

Two independent checks of whether dividing out C recovers the true reflection:
  (1) Silicon: known THz index n~3.418, k~0. Which estimator hits it?
  (2) CNT-0deg vs CNT-90deg: SAME film, so true r_back is identical. Which
      estimator makes the two rotations agree?
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

def build():
    ds = DataSet(DATA, config=config)
    ds.load_all_data()
    thz.build_full_trace_reflection(ds)
    ds.group_files(keywords=['type'])
    thz.subtract_baseline(ds)
    thz.define_reflection_regions(ds, config)
    thz.window_pulses_fixed_width(ds, half_width_ps=3.0, show_graph=False)
    nfft = config['fft']['n_fft']
    thz.fft_spectrum(ds, segment='second_reflection', n_fft=nfft)
    thz.fft_spectrum(ds, segment='first_reflection', n_fft=nfft)
    thz.transfer_function(ds, ref_type='reference')
    return ds

def invert_current(ds):
    """Run reflection inversion on whatever transfer_H currently is; return per-sample n,k,f."""
    thz.invert_nk_reflection(ds, geometry='window',
        theta_deg=45.0, polarization='s', n_window=1.95)
    out = {}
    for fname, obj in ds.data.items():
        if ds.data.is_reference(fname):
            continue
        p = obj.processing_dict
        out[fname] = (p.get('fft_freq'), p.get('n'), p.get('k'))
    return out

# --- estimator A: self-referenced H ---
ds = build()
res_H = invert_current(ds)

# --- estimator B: H/C = Y2s/Y2r. Overwrite transfer_H with H/C, re-invert. ---
ds2 = build()
for fname, obj in ds2.data.items():
    if ds2.data.is_reference(fname):
        continue
    p = obj.processing_dict
    H = p.get('transfer_H'); C = p.get('selfref_correction')
    if H is not None and C is not None:
        p['transfer_H'] = H / C
res_HC = invert_current(ds2)

def band_median(f, y, lo=0.4, hi=1.5):
    m = (f*1e-12 >= lo) & (f*1e-12 <= hi) & np.isfinite(y)
    return np.nanmedian(y[m]) if m.any() else np.nan

print("\n=== n (median 0.4-1.5 THz) : H (self-ref) vs H/C (=Y2s/Y2r) ===")
print(f"{'sample':40s} {'n[H]':>8s} {'k[H]':>8s} {'n[H/C]':>8s} {'k[H/C]':>8s}")
for fname in res_H:
    f, nH, kH = res_H[fname]
    _, nC, kC = res_HC[fname]
    print(f"{fname[:40]:40s} {band_median(f,nH):8.3f} {band_median(f,kH):8.3f} "
          f"{band_median(f,nC):8.3f} {band_median(f,kC):8.3f}")

# --- agreement metric: CNT-0 vs CNT-90 (same film) ---
def get(res, key):
    for fn in res:
        if key in fn: return res[fn]
    return None
def disagree(a, b, lo=0.4, hi=1.5):
    fa, na, ka = a; fb, nb, kb = b
    m = (fa*1e-12>=lo)&(fa*1e-12<=hi)&np.isfinite(na)&np.isfinite(nb)
    return np.nanmean(np.abs(na[m]-nb[m]))
c0H, c90H = get(res_H,'CNT-0'), get(res_H,'CNT-90')
c0C, c90C = get(res_HC,'CNT-0'), get(res_HC,'CNT-90')
print("\n=== CNT-0 vs CNT-90 disagreement in n (same film -> should be ~0) ===")
print(f"  self-ref  H : mean |dn| = {disagree(c0H,c90H):.3f}")
print(f"  H/C=Y2s/Y2r : mean |dn| = {disagree(c0C,c90C):.3f}")

# --- plot n for all, both estimators ---
fig, axes = plt.subplots(1, 2, figsize=(14,5), layout='constrained')
fig.suptitle('Reflection n: self-ref H (solid) vs H/C=Y2s/Y2r (dashed)')
for fname in res_H:
    f, nH, _ = res_H[fname]; _, nC, _ = res_HC[fname]
    fthz=f*1e-12; b=(fthz>=0.3)&(fthz<=2.0)
    l=axes[0].plot(fthz[b], nH[b], label=fname)[0]
    axes[1].plot(fthz[b], nC[b], '--', color=l.get_color(), label=fname)
axes[0].set_title('n from self-ref H'); axes[1].set_title('n from H/C = Y2s/Y2r')
for ax in axes:
    ax.axhline(3.418, color='k', ls=':', lw=0.8, label='Si n=3.418')
    ax.set_xlabel('Frequency (THz)'); ax.set_ylabel('n'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.savefig(os.path.join(OUT,'audit_C_inversion_benchmark.png'), dpi=110)
print("\nsaved audit_C_inversion_benchmark.png")
