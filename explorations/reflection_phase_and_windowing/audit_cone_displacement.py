"""Audit the cone claim: moving the reflection plane ~1mm (gold mirror back-face ->
front-face position) -- does it reshape the sampled frequency cone?

Robust cone  => transfer H=front/back is flat in |.| (no frequency reshaping), pure
linear phase (geometric delay). Self-contained: window each pulse around its own peak,
FFT on a shared grid, ratio. No interactive centering.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
REPO=r'C:\Users\Samuel\matchbook\thz'; sys.path.insert(0,REPO)
from dataset_core import DataSet
OUT=os.path.join(REPO,'explorations','reflection_phase_and_windowing')
DATA=r'C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\cone_tests'

ds=DataSet(DATA); ds.load_all_data(case_insensitive=True, explicit_dir=True)
traces={}
for fn,o in ds.data.items():
    d=o.data                      # averaged (N,2): time[s], amp
    traces[fn]=(d[:,0], d[:,1])
    pk=d[np.argmax(np.abs(d[:,1])),0]
    print(f"{fn:55s} n={d.shape[0]:4d} dt={np.median(np.diff(d[:,0]))*1e12:.4f}ps peak@{pk*1e12:.2f}ps")

HALF_PS=4.0; NFFT=4096
def windowed_fft(t,y):
    dt=np.median(np.diff(t)); half=int(round(HALF_PS*1e-12/dt))
    pk=int(np.argmax(np.abs(y))); w=tukey(2*half+1,alpha=0.3)
    seg=np.zeros_like(y); lo=max(pk-half,0); hi=min(pk+half+1,y.size)
    seg[lo:hi]=y[lo:hi]*w[lo-(pk-half):hi-(pk-half)]
    Y=np.fft.rfft(seg,n=NFFT); f=np.fft.rfftfreq(NFFT,dt)
    Y=Y*np.exp(-2j*np.pi*f*t[0])         # absolute-time phase ref (like the pipeline)
    return f,Y
front=[fn for fn in traces if 'front' in fn][0]
back=[fn for fn in traces if 'back' in fn][0]
f,Yf=windowed_fft(*traces[front]); _,Yb=windowed_fft(*traces[back])
H=Yf/Yb
fthz=f*1e-12; b=(fthz>=0.3)&(fthz<=2.0)&np.isfinite(H)
mag=np.abs(H)[b]
print(f"\n|H| front/back over 0.3-2.0 THz: median={np.nanmedian(mag):.3f} "
      f"std/med={np.nanstd(mag)/np.nanmedian(mag):.1%}")
ph=np.unwrap(np.angle(H[b])); coef=np.polyfit(f[b],ph,1)
print(f"phase slope -> delay={coef[0]/(2*np.pi)*1e12:+.3f} ps")
resid=ph-np.polyval(coef,f[b])
print(f"phase residual after linear fit: std={np.nanstd(resid):.3f} rad")
fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
fig.suptitle('Cone displacement audit: gold front-face vs back-face (~1mm reflection-plane move)')
for fn,(t,y) in traces.items():
    axes[0].plot(t*1e12, y/np.max(np.abs(y)), label=fn, lw=1)
axes[0].set_title('raw traces (norm)'); axes[0].set_xlabel('time (ps)'); axes[0].legend(fontsize=7)
axes[1].plot(fthz[b], np.abs(H)[b]); axes[1].axhline(np.nanmedian(mag),color='k',ls=':',lw=0.8)
axes[1].set_title('|H| front/back (flat=cone robust)'); axes[1].set_xlabel('THz'); axes[1].set_ylim(0,2)
axes[2].plot(fthz[b], ph, label='unwrapped'); axes[2].plot(fthz[b], np.polyval(coef,f[b]),'--',label='linear fit')
axes[2].set_title('phase (linear=pure delay)'); axes[2].set_xlabel('THz'); axes[2].legend(fontsize=8)
for ax in axes: ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT,'audit_cone_displacement.png'),dpi=110)
print("\nsaved audit_cone_displacement.png")
