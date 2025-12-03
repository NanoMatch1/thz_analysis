# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:30:56 2023

@author: Marco Ballabio

References: Jepsen, PU (2019)       https://doi.org/10.1007/s10762-019-00578-0
            Pupeza, I  (2007)       https://doi.org/10.1364/OE.15.004335
            Vázquez-Cabo, J (2016)  http://dx.doi.org/10.1016/j.optcom.2015.12.069
            
uses ε = ε1 + iε2
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants as cs 
from scipy.signal import hilbert
import os
from math import e

#all the following imports will end up in a single module
import dataimport as di
import fft_err
import windowing as w
import padding as pad
import phase_interpolation as phi

def compare_windowing(comparison_dict, show_ref=False):
    '''Compares different windowing/padding methods by plotting the time and frequency domain traces.'''

    raw_data = comparison_dict.pop('raw', {})

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for index, (method, datasets) in enumerate(comparison_dict.items()):
        for label, data in datasets.items():
            if show_ref is False and label.lower() == 'reference':
                continue
            ax[index].plot(data['Time (ps)'], data['Mean'], label=f'{label}')
            ax[index].plot(raw_data[label]['Time (ps)'], raw_data[label]['Mean'], linestyle='--', alpha=0.5, label=f'{label} (raw)')
        ax[index].set_title(f'Time Domain - {method}')
        ax[index].set_xlabel('Time (ps)')
        ax[index].set_ylabel('Amplitude (V)')
        ax[index].legend()
    
    plt.show()

def synth_thz_pulse(t, t0=0, width=0.5, f0=1.0):
    envelope = np.exp(-((t - t0)/width)**2)
    carrier   = np.cos(2*np.pi*f0*(t - t0))
    return envelope * carrier

def complex_synthetic_pulse():
    '''Generates a synthetic THz pulse in the frequency domain and transforms it to the time domain.'''
    

    # ------------- frequency domain specification -------------
    f_max = 5.0          # THz
    N = 512             # must be power of 2 for best IFFT
    df = f_max / N
    freq = np.linspace(0, f_max, N)

    # amplitude spectrum (broadband)
    A = np.exp(-(freq - 1.0)**2 / (1.0**2))  # ~1 THz center, bandwidth ~2 THz

    # smooth random phase (to mimic real optics)
    rng = np.random.default_rng(42)
    phi = np.cumsum(rng.normal(scale=0.01, size=N))   # low jitter random walk
    phi -= phi[0]

    # complex frequency spectrum
    E_f = A * np.exp(1j * phi)

    # ------------- time domain -------------
    # IFFT
    E_t = np.fft.irfft(E_f, n=N*2)

    # time axis (ps)
    dt = 1 / (2 * f_max)        # THz → ps: dt = 1/(2*f_max)
    t = np.arange(len(E_t)) * dt

    # create delayed sample
    delay_ps = 2.5
    delay_pts = int(delay_ps / dt)
    # E_t_delayed = np.roll(E_t, delay_pts)

    # ------------- Package into DataFrames -------------
    ref_df = pd.DataFrame({'Time (ps)': t, 'Mean': E_t, 'std error': 0.01})
    # sam_df = pd.DataFrame({'Time (ps)': t, 'Mean': E_t_delayed, 'std error': 0.01})

    plt.plot(ref_df['Time (ps)'], ref_df['Mean'], label='Reference')
    # plt.plot(sam_df['Time (ps)'], sam_df['Mean'], label='Sample')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude (V)')
    plt.title('Synthetic Signals from Frequency Domain')
    plt.legend()
    plt.show()


# 2) Build a sample trace delayed by 2.5 ps

# %% Initialization

# filepath = r'C:\Users\Usuario\Desktop\Spintronics\28Nov2024'
# reference = 'air_ref.acc'
# sample = 'SrTiO3_singlepolished.acc'

# filepath = r'Z:\Users\Marco\29jul2025\powder'
# reference = 'TDS_day3_sandwich_ref.acc'
# sample = 'TDS_NiHITP_day3.acc'

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# filepath = r'C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP'
reference = ['reference', '300K']
sample = ['sample', '300K']


samples = [file for file in os.listdir(filepath) if file.endswith('.acc') and all(item in file for item in sample)]
references = [file for file in os.listdir(filepath) if file.endswith('.acc') and all(item in file for item in reference)]


sample = samples[0]
reference = references[0]

thickness = 2e-4
eps_inf = 1 # glass 3.42, Si 11.6
ns = 1.9 #substate n
Z0 = cs.physical_constants['characteristic impedance of vacuum'][0]


def simulate_test():
    t = np.linspace(-5, 20, 5012)
    ref_synth = synth_thz_pulse(t, t0=2, width=0.5, f0=1.0)
    sam_synth = synth_thz_pulse(t, t0=4.5, width=0.5, f0=1.0)

    ref_synth_df = pd.DataFrame({'Time (ps)': t, 'Mean': ref_synth, 'std error': 0.01})
    sam_synth_df = pd.DataFrame({'Time (ps)': t, 'Mean': sam_synth, 'std error': 0.01})

    plt.plot(ref_synth_df['Time (ps)'], ref_synth_df['Mean'], label='Reference')
    plt.plot(sam_synth_df['Time (ps)'], sam_synth_df['Mean'], label='Sample')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude (V)')
    plt.title('Synthetic Reference Signal')
    plt.show()

    #Fourier Transform
    # ref_fft = fft_err.fft_err(ref_synth_df)
    # sam_fft = fft_err.fft_err(sam_synth_df)
    ref_fft = fft_err.fft_err_simple(ref_synth_df)
    sam_fft = fft_err.fft_err_simple(sam_synth_df)

    # calculates initial phase offset
    tref = ref_synth_df.iloc[np.argmax(abs(ref_synth_df['Mean'])),0]
    tsam = sam_synth_df.iloc[np.argmax(abs(sam_synth_df['Mean'])),0]

    min_time = min(t)
    max_time = max(t)
    print(f"Time window: {min_time} ps to {max_time} ps"
        )
    delta_t_time_ps = tsam - tref
    print("Time-domain delay:", delta_t_time_ps, "ps")

    phioffset = phi.phaseoffset(ref_synth_df, sam_synth_df) #to account for different time windows starts

    phidifference, delta_t_phase_synth = phi.phaseex_v2(ref_fft, sam_fft, show_graph=True)

    # phidifference_edge, delta_t_phase_ps_edge = phi.phaseex_v2(ref_edge_fft, sam_edge_fft, show_graph=True)

    print("Phase difference delay synthetic, no window:", delta_t_phase_synth, "ps")

    breakpoint()

# simulate_test()

ref_t = di.dataimport(filepath,reference)
sam_t = di.dataimport(filepath,sample)
# --> Plug in from new dataimport module

# breakpoint()
# %% Analysis

# moves the pulse to the center an pad 0 at the edges
# import matplotlib.pyplot as plt
# print("Filename:", reference    )ref_centered = centered_results['Reference']
ref_centered = pad.centerpad(ref_t, length_factor=3)
sam_centered = pad.centerpad(sam_t, length_factor=3)

ref_edge_windowed = pad.edge_window_pad(ref_t, alpha=0.4, padding=True, padding_factor=1)
sam_edge_windowed = pad.edge_window_pad(sam_t, alpha=0.4, padding=True, padding_factor=1)

# TODO: Multiple queries.
# 1. Why does linearly increasing the padding factor seem to linearly increase the phase difference in when using the simple FFT method?
# 2. I can see a dispersive shape to the phase when using the informed phase unwrapping for the sample. it looks real, is this the effective refractive inedex of the sample?
# 3. Even when the informed phase unwrapping is used and the dispersive shape is there and seems real, I still dont get the correct phase delay in ps when comparing to the time domain delay. Why?
# 4. I only get the shape when I have no padding or very little padding. Why?


comparison = {'centerpadded': 
              {'reference': ref_centered, 'sample': sam_centered}, 
              'edge_windowed': 
              {'reference': ref_edge_windowed, 'sample': sam_edge_windowed},
              'raw': {'reference': ref_t, 'sample': sam_t}}

compare_windowing(comparison)

#Fourier Transform

ref = fft_err.fft_err(ref_t)
sam = fft_err.fft_err(sam_t)

refw = fft_err.fft_err(ref_centered)
samw = fft_err.fft_err(sam_centered)
ref_edge_fft = fft_err.fft_err(ref_edge_windowed)
sam_edge_fft = fft_err.fft_err(sam_edge_windowed)
# ref_edge_fft = fft_err.fft_err_simple(ref_edge_windowed)
# sam_edge_fft = fft_err.fft_err_simple(sam_edge_windowed)


#SNR (noise calculated between 4-10 THz for ZnTe)

lo_threshold = 4
up_threshold = 10
nrange = ref.loc[ref['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
nfloor_ref = nrange.mean()

DR = np.max(ref['Amplitude'].values)/nfloor_ref

nrangew = refw.loc[refw['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
nfloor_refw = nrangew.mean()

DRw = np.max(refw['Amplitude'].values)/nfloor_refw

DRimpr = 10*np.log10(DRw/DR)

# calculates initial phase offset
tref = ref_t.iloc[np.argmax(abs(ref_t['Mean'])),0]
tsam = sam_t.iloc[np.argmax(abs(sam_t['Mean'])),0]

delta_t_time_ps = tsam - tref
print("Time-domain delay:", delta_t_time_ps, "ps")


phiref = 2*np.pi*samw['Frequency (THz)']*(tref)
phisam = 2*np.pi*samw['Frequency (THz)']*(tsam)

phidiff = 2*np.pi*samw['Frequency (THz)']*(tsam-tref)

# phioffset = phi.phaseoffset(ref_centered, sam_centered) #to account for different time windows starts
# phioffset_edge = phi.phaseoffset(ref_edge_windowed, sam_edge_windowed) #to account for different time windows starts

phidifference, delta_t_phase_ps = phi.phaseex_v2(refw, samw, show_graph=True)

phidifference_edge, delta_t_phase_ps_edge = phi.phaseex_v2(ref_edge_fft, sam_edge_fft, show_graph=True)

print("Phase difference delay (windowed):", delta_t_phase_ps, "ps")
print("Phase difference delay (edge windowed):", delta_t_phase_ps_edge, "ps")

breakpoint()

print("Phase-domain delay:", delta_t_phase_ps, "ps")
print("Phase offset difference (time - phase):", delta_t_time_ps - delta_t_phase_ps, "ps")

if abs(abs(delta_t_phase_ps) - abs(delta_t_time_ps)) > 1:
    raise ValueError("Significant discrepancy between time-domain and phase-domain delays.")


#transfer function
breakpoint()
T = fft_err.transfer_function(refw, samw, phioffset)
# %% self standing film approximation

#guesses refractive index with thin, self standing film approximation
nguess = 1+((phidifference-phioffset)*cs.c)/(2*np.pi*samw['Frequency (THz)']*1e12*thickness)
kguess = -cs.c/(2*np.pi*thickness*refw['Frequency (THz)']*1e12)*np.log(((nguess+1)**2/4/nguess)*(samw['Amplitude']/refw['Amplitude']))

#Gouy shift - gaussian beam correction - Kužel2010 https://doi.org/10.1364/OE.18.015338
# beta = cs.c**2/(2*(samw['Frequency (THz)']*1e12)**2*np.pi**2*(1e-3)**2)
# nguess = nguess-beta*((nguess-1)/nguess)

#calculates complex permittivity
eps1 = nguess**2-kguess**2
eps2 = 2*nguess*kguess

#loss tangent
losstg =eps2/eps1

#calculates complex conductivity
realc = 4*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*nguess*kguess
imagc = 2*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*(eps_inf-nguess**2+kguess**2)


# epsh = eps_inf - (h1.imag/(2*np.pi*refw['Frequency (THz)']*1e12)/cs.epsilon_0)
# nh = np.sqrt(epsh+np.sqrt(epsh**2+(realc/(2*np.pi*refw['Frequency (THz)']*1e12)/cs.epsilon_0)**2)/2)

# %% thin conductive film

#somehow it returns the conjugate of what expected for n_squared, still invesigating --> sign convention epsilon

# n_squared = 1j*cs.c*((1+ns)/(T['Amplitude']*e**(1j*T['Phase']))-1-ns)/(2*np.pi*T['Frequency (THz)']*1e12*thickness)
# n_squared=n_squared.to_numpy()
# eps1 = pd.Series(n_squared.real)
# eps2 = pd.Series(n_squared.imag)
# nguess = pd.Series(np.sqrt(0.5*(eps1+np.sqrt(eps1**2+eps2**2))))
# kguess = pd.Series(np.sqrt(0.5*(np.sqrt(eps1**2+eps2**2)-eps1)))
# realc = -2*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*n_squared.imag
# imagc = 2*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*(eps_inf-n_squared.real)

# %% sandwich
# https://doi.org/10.1364/OE.510393 Novelli(2024)

nguess = 1+((phidifference-phioffset)*cs.c)/(2*np.pi*samw['Frequency (THz)']*1e12*thickness)
kguess = -cs.c/(2*np.pi*thickness*refw['Frequency (THz)']*1e12)*np.log(((nguess+ns)**2/(1+ns)**2/nguess)*(samw['Amplitude']/refw['Amplitude']))

#calculates complex permittivity
eps1 = nguess**2-kguess**2
eps2 = 2*nguess*kguess

#loss tangent
losstg =eps2/eps1

#calculates complex conductivity
realc = 4*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*nguess*kguess
imagc = 2*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*(eps_inf-nguess**2+kguess**2)

# %% Plots

realc.fillna(0,inplace = True)
# eps1.fillna(0,inplace = True)
h1 = hilbert(realc.values)

plt.figure(0,dpi=300)
plt.plot(ref_t['Time (ps)'],ref_t['Mean'],label='Reference')
plt.plot(sam_t['Time (ps)'],sam_t['Mean'],label='Sample')
plt.plot(ref_centered['Time (ps)'],ref_centered['Mean'],label='Reference windowed', linestyle='-')
plt.plot(sam_centered['Time (ps)'],sam_centered['Mean'],label='Sample windowed')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude (V)')
plt.xlim(left=ref_t['Time (ps)'].iloc[0],right=sam_t['Time (ps)'].iloc[-1])
plt.legend()


plt.figure(1)
plt.plot(ref['Frequency (THz)'],ref['Amplitude'],label='Reference')
plt.plot(refw['Frequency (THz)'],refw['Amplitude'],label='Reference windowed')
plt.plot(sam['Frequency (THz)'],sam['Amplitude'],label='Sample')
plt.plot(samw['Frequency (THz)'],samw['Amplitude'],label='Sample windowed')
plt.legend()
plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.xlim(0,10)
plt.yscale('log')

plt.figure(11)
plt.plot(ref['Frequency (THz)'],ref['Δ(Amplitude)']/ref['Amplitude'],label='Reference')
plt.plot(refw['Frequency (THz)'],refw['Δ(Amplitude)']/refw['Amplitude'],label='Reference windowed')
# plt.plot(sam['Frequency (THz)'],sam['Δ(Amplitude)'],label='Sample')
# plt.plot(samw['Frequency (THz)'],samw['Δ(Amplitude)'],label='Sample windowed')
plt.legend()
# plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.xlim(0,3)
plt.ylim(0,0.51)

plt.figure(10)
plt.title('Transfer function')
plt.plot(T['Frequency (THz)'],T['Amplitude'],label='Amplitude')
plt.legend()
plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.xlim(0,4)
plt.ylim(0.2,1.1)
# plt.yscale('log')

#make dB scale
plt.figure(2)
plt.plot(ref['Frequency (THz)'],10*np.log10(ref['Amplitude']/np.max(ref['Amplitude'].values)),label='Reference')
plt.plot(refw['Frequency (THz)'],10*np.log10(refw['Amplitude']/np.max(refw['Amplitude'].values)),label='Reference windowed')
plt.plot(sam['Frequency (THz)'],10*np.log10(sam['Amplitude']/np.max(ref['Amplitude'].values)),label='Sample')
plt.plot(samw['Frequency (THz)'],10*np.log10(samw['Amplitude']/np.max(refw['Amplitude'].values)),label='Sample windowed')
# plt.axhline(y=10*np.log10(nfloor_ref/np.max(ref['Amplitude'].values)), linestyle = 'dashed')
# plt.axhline(y=10*np.log10(nfloor_refw/np.max(refw['Amplitude'].values)), linestyle = 'dashed')
plt.legend()
plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.ylabel('dB')
plt.xlim(0,10)
# plt.yscale('log')

plt.figure(3)
plt.plot(T['Frequency (THz)'],T['Phase'], label='transfer $\phi$' )
plt.plot(refw['Frequency (THz)'],phidifference-phioffset, label='$\Delta \phi$',linestyle='dashed' )

# plt.xlim(0.0,4)
plt.ylim(-1,100)
plt.legend()


fig, (ax1,ax2)=plt.subplots(2,1,sharex=True)
fig.suptitle('Refractive index')
plt.xlim(0,4)
plt.subplot(211)
plt.ylabel('n')
plt.plot(refw['Frequency (THz)'], nguess)
plt.ylim(0.9,1.2)
# plt.autoscale(axis='y')

plt.tick_params(axis='y')

plt.subplot(212)
plt.ylabel('k')
plt.plot(refw['Frequency (THz)'], kguess, color='crimson')
plt.axhline(0,linestyle = 'dashed', color ='gray', lw = '0.5')
plt.ylim(-0.11,1)
# plt.autoscale(axis='y')
plt.xlabel('Frequency (THz)')
plt.show()


plt.figure(4)
plt.title('Permittivity')
plt.plot(refw['Frequency (THz)'],eps1,label='$\epsilon_1$')
plt.plot(refw['Frequency (THz)'],eps2,label='$\epsilon_2$')
plt.axhline(0,linestyle = 'dashed', color ='gray', lw = '0.5')
plt.xlim(0.0,4)
plt.ylim(-2,5)
# plt.autoscale(axis='y')
plt.xlabel('Frequency (THz)')
plt.legend()
# plt.ylim(3,5)



plt.figure(5)
plt.title('Conductivity (S/m)')
plt.plot(refw['Frequency (THz)'],realc,label='$\sigma_1$')
plt.plot(refw['Frequency (THz)'],imagc,label='$\sigma_2$')
plt.axhline(0,linestyle = 'dashed', color ='gray', lw = '0.5')
# plt.plot(refw['Frequency (THz)'],h1.imag,label='$\sigma_2$ hilbert')
# plt.ylim(0,500)
plt.xlim(0.5,2.5)
plt.ylim(-50 ,100)
# plt.autoscale(axis='y')
plt.xlabel('Frequency (THz)')
plt.legend()

plt.show()

# %% Savefile

# # Ask the user if they want to export the DataFrame
# export_decision = input("Do you want to save the results? (Y/n): ")
# savedir = 'Results'
# filename = sample.split(".")[0]

# path = os.path.join(filepath,savedir)
# if not os.path.exists(path):
#     os.mkdir(path)

# if export_decision.lower() == 'y':
    
#     #create time-domain data file
#     timedata = pd.concat([ref_t,ref_centered,sam_t,sam_centered],
#                           axis=1, keys=['Reference','windowed Reference', 'Sample', 'windowed Sample'],
#                           names=['Dataset','Quantity'])   
#     freqdata = pd.concat([refw,samw],
#                           axis=1,keys=['Reference','Sample'],
#                           names=['Dataset'])
#     optidata = pd.concat([refw['Frequency (THz)'], nguess,kguess,eps1,eps2,realc,imagc],
#                           axis=1,keys=['Frequency','n','k','ε1','ε2','σRe','σIm'])
#     print(f'ε∞ = {eps_inf}')
    
#     # Ask for the destination folder
#     destination_folder = path
    
#     # Specify the separator
#     separator = '\t'

#     # # Export the DataFrame to a text file in the specified destination folder (fix for nix systems)
#     file_path = f"{destination_folder}\\{filename}_timeoutput.txt"
#     timedata.to_csv(file_path, sep=separator, index=False)
#     file_path = f"{destination_folder}\\{filename}_frequencyoutput.txt"
#     freqdata.to_csv(file_path, sep=separator, index=False)
#     file_path = f"{destination_folder}\\{filename}_opticalconstants.txt"
#     optidata.to_csv(file_path, sep=separator, index=False)

#     print(f"Results exported to {file_path}")
# else:
#     print("Results not saved.")