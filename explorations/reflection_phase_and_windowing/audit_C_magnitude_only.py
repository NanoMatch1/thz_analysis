"""Third estimator: H/|C| (magnitude-only de-distortion).

H        : self-ref, cancels timing, but front-spot AMP distortion in Y1s leaks in.
H/C      : = Y2s/Y2r, removes front distortion AND timing cancellation (re-injects
           absolute back-pulse timing offset -> wrong n when timing differs).
H/|C|    : removes only the front-spot AMPLITUDE distortion, keeps H's phase (so the
           inter-pulse timing cancellation that makes self-ref work is preserved).
Benchmark: Si n~3.418; CNT-0 vs CNT-90 must agree (same film).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO = r'C:\Users\Samuel\matchbook\thz'; sys.path.insert(0, REPO)
from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
DATA = r'C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\main_alignment_tests'
OUT = os.path.join(REPO,'explorations','reflection_phase_and_windowing')
config = {
    "general":{"show_graph":False},
    "geometry":{"theta_external_deg":45.0,"polarization":"s","n_sio2":1.95},
    "regions":{"first_reflection":(152.5,159),"second_reflection":(177,183.8)},
    "centering":{"mode":"crop"},"window":{"type":"hann","alpha":1.0,"half_width_ps":3.0},
    "pad":{"extend_factor":1.0},"fft":{"norm":"backward","amplitude_scale":1.0,"n_fft":2000},
    "transfer":{"self_reference":True,"apply_snr_mask":True,"min_ref_amp_rel":1e-3,
                "regularization_eps":1e-30,"unwrap_phase":True},
    "mask":{"snr_thresh_db":10,"tail_fraction":0.25,"min_contiguous_bins":3},
    "derive":{"eps_background":11.7},
}
def build():
    ds=DataSet(DATA,config=config); ds.load_all_data()
    thz.build_full_trace_reflection(ds); ds.group_files(keywords=['type'])
    thz.subtract_baseline(ds); thz.define_reflection_regions(ds,config)
    thz.window_pulses_fixed_width(ds,half_width_ps=3.0,show_graph=False)
    thz.fft_spectrum(ds,segment='second_reflection',n_fft=2000)
    thz.fft_spectrum(ds,segment='first_reflection',n_fft=2000)
    thz.transfer_function(ds,ref_type='reference'); return ds
def invert(ds,transform):
    for fn,obj in ds.data.items():
        if ds.data.is_reference(fn): continue
        p=obj.processing_dict; H=p.get('transfer_H'); C=p.get('selfref_correction')
        if H is not None and C is not None: p['transfer_H']=transform(H,C)
    thz.invert_nk_reflection(ds,geometry='window',theta_deg=45.0,polarization='s',n_window=1.95)
    return {fn:(o.processing_dict.get('fft_freq'),o.processing_dict.get('n'),o.processing_dict.get('k'))
            for fn,o in ds.data.items() if not ds.data.is_reference(fn)}
estimators={'H':lambda H,C:H,'H/C':lambda H,C:H/C,'H/|C|':lambda H,C:H/np.abs(C)}
results={name:invert(build(),tf) for name,tf in estimators.items()}
def bmed(f,y,lo=0.4,hi=1.5):
    m=(f*1e-12>=lo)&(f*1e-12<=hi)&np.isfinite(y); return np.nanmedian(y[m]) if m.any() else np.nan
def get(res,key):
    for fn in res:
        if key in fn: return res[fn]
def dis(a,b,lo=0.4,hi=1.5):
    fa,na,_=a; fb,nb,_=b; m=(fa*1e-12>=lo)&(fa*1e-12<=hi)&np.isfinite(na)&np.isfinite(nb)
    return np.nanmean(np.abs(na[m]-nb[m]))
print(f"\n{'estimator':8s} | {'Si n':>7s} (true 3.418) | {'CNT0 n':>7s} {'CNT90 n':>7s} | {'|dn| CNT0-90':>12s}")
for name in estimators:
    r=results[name]; si=get(r,'Si'); c0=get(r,'CNT-0'); c90=get(r,'CNT-90')
    print(f"{name:8s} | {bmed(*si):7.3f}            | {bmed(*c0):7.3f} {bmed(*c90):7.3f} | {dis(c0,c90):12.3f}")
# plot Si n for the three estimators
fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
for name in estimators:
    f,n,_=get(results[name],'Si'); fthz=f*1e-12; b=(fthz>=0.3)&(fthz<=2.0)
    ax.plot(fthz[b],n[b],label=f'Si {name}')
ax.axhline(3.418,color='k',ls=':',label='Si n=3.418'); ax.set_ylim(0,6)
ax.set_xlabel('Frequency (THz)'); ax.set_ylabel('n'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Silicon n: which estimator hits 3.418?')
fig.savefig(os.path.join(OUT,'audit_C_magnitude_only.png'),dpi=110)
print("\nsaved audit_C_magnitude_only.png")
