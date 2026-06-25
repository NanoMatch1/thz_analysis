"""Principled C-correction: divide by the DISTORTION part of C, keep the timing part.

C = Y1r/Y1s = T(f) * g(f)  where  T = exp(2pi i f * dt_front)  is the pure front-pulse
timing offset (a linear phase ramp) and g is the front-spot distortion we want to remove.
self-ref H already cancels timing, so dividing by the FULL C re-injects T (breaks Si).
Dividing by C_detrended = C/T removes only g.  Estimators compared:
  H, H/C, H/|C|, H/C_detrended.
Metrics: (1) Si n vs 3.418; (2) complex-transfer agreement CNT-0 vs CNT-90 (same film):
  mean |H0 - H90| / mean |H90| over 0.3-1.5 THz  (inversion-free, air-gap-robust).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=r'C:\Users\Samuel\matchbook\thz'; sys.path.insert(0,REPO)
from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
DATA=r'C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\main_alignment_tests'
OUT=os.path.join(REPO,'explorations','reflection_phase_and_windowing')
config={"general":{"show_graph":False},
 "geometry":{"theta_external_deg":45.0,"polarization":"s","n_sio2":1.95},
 "regions":{"first_reflection":(152.5,159),"second_reflection":(177,183.8)},
 "centering":{"mode":"crop"},"window":{"type":"hann","alpha":1.0,"half_width_ps":3.0},
 "pad":{"extend_factor":1.0},"fft":{"norm":"backward","amplitude_scale":1.0,"n_fft":2000},
 "transfer":{"self_reference":True,"apply_snr_mask":True,"min_ref_amp_rel":1e-3,
             "regularization_eps":1e-30,"unwrap_phase":True},
 "mask":{"snr_thresh_db":10,"tail_fraction":0.25,"min_contiguous_bins":3},
 "derive":{"eps_background":11.7}}
def build():
    ds=DataSet(DATA,config=config); ds.load_all_data()
    thz.build_full_trace_reflection(ds); ds.group_files(keywords=['type'])
    thz.subtract_baseline(ds); thz.define_reflection_regions(ds,config)
    thz.window_pulses_fixed_width(ds,half_width_ps=3.0,show_graph=False)
    thz.fft_spectrum(ds,segment='second_reflection',n_fft=2000)
    thz.fft_spectrum(ds,segment='first_reflection',n_fft=2000)
    thz.transfer_function(ds,ref_type='reference'); return ds

def detrend_phase(f, C, lo=0.3, hi=1.5):
    """Remove the best-fit linear phase (timing ramp) from C over the trusted band."""
    m=(f*1e-12>=lo)&(f*1e-12<=hi)&np.isfinite(C)
    ph=np.unwrap(np.angle(C[m]))
    slope=np.polyfit(f[m],ph,1)[0]              # rad per Hz  -> dt_front = slope/2pi
    return C*np.exp(-1j*slope*f), slope/(2*np.pi)

estimators={
 'H'           : lambda H,C,f: H,
 'H/C'         : lambda H,C,f: H/C,
 'H/|C|'       : lambda H,C,f: H/np.abs(C),
 'H/C_detrend' : lambda H,C,f: H/detrend_phase(f,C)[0],
}
# capture H per sample per estimator (pre-inversion) + invert for Si n
Hcap={name:{} for name in estimators}
res={name:{} for name in estimators}
for name,tf in estimators.items():
    ds=build()
    for fn,o in ds.data.items():
        if ds.data.is_reference(fn): continue
        p=o.processing_dict; H=p.get('transfer_H'); C=p.get('selfref_correction'); f=p.get('fft_freq')
        Hc=tf(H,C,f); Hcap[name][fn]=(f,Hc); p['transfer_H']=Hc
    thz.invert_nk_reflection(ds,geometry='window',theta_deg=45.0,polarization='s',n_window=1.95)
    for fn,o in ds.data.items():
        if ds.data.is_reference(fn): continue
        res[name][fn]=(o.processing_dict.get('fft_freq'),o.processing_dict.get('n'))

def find(d,key):
    for fn in d:
        if key in fn: return d[fn]
def si_n(name,lo=0.4,hi=1.5):
    f,n=find(res[name],'Si'); m=(f*1e-12>=lo)&(f*1e-12<=hi)&np.isfinite(n); return np.nanmedian(n[m])
def cnt_transfer_disagree(name,lo=0.3,hi=1.5):
    f0,H0=find(Hcap[name],'CNT-0'); f9,H9=find(Hcap[name],'CNT-90')
    m=(f0*1e-12>=lo)&(f0*1e-12<=hi)&np.isfinite(H0)&np.isfinite(H9)
    return np.nanmean(np.abs(H0[m]-H9[m]))/np.nanmean(np.abs(H9[m]))

print(f"\n{'estimator':12s} | {'Si n':>7s} (3.418) | {'CNT0-90 transfer disagree (rel)':>32s}")
for name in estimators:
    print(f"{name:12s} | {si_n(name):7.3f}        | {cnt_transfer_disagree(name):32.3f}")
# report fitted front timing offset
ds=build()
print("\nfront-pulse timing offset dt_front (C linear-phase fit):")
for fn,o in ds.data.items():
    if ds.data.is_reference(fn): continue
    p=o.processing_dict; _,dt=detrend_phase(p['fft_freq'],p['selfref_correction'])
    print(f"  {fn[:42]:42s} dt_front = {dt*1e12:+.3f} ps")
