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

# %% Initialization

# filepath = r'C:\Users\Usuario\Desktop\Spintronics\28Nov2024'
# reference = 'air_ref.acc'
# sample = 'SrTiO3_singlepolished.acc'

# filepath = r'Z:\Users\Marco\29jul2025\powder'
# reference = 'TDS_day3_sandwich_ref.acc'
# sample = 'TDS_NiHITP_day3.acc'

filepath = r'C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP'
reference = ['reference', '300K']
sample = ['sample', '300K']


samples = [file for file in os.listdir(filepath) if file.endswith('.acc') and all(item in file for item in sample)]
references = [file for file in os.listdir(filepath) if file.endswith('.acc') and all(item in file for item in reference)]


# filepath = r'Z:\Users\Marco\04jul2025'
# reference = '1mm_mask_ref.acc'
# sample = 'Coronado_Mof_1mmmask_270deg.acc'

# filepath = r'C:\Users\Usuario\Desktop\MOF\SCosta\16nov2023'
# reference = 'ref_rt_vacuum.acc'
# sample = 'FeMOF_hex_350_2.acc'

# filepath = r'C:\Users\Usuario\Desktop\Chris\Silicon cryo\marzo24'
# reference = 'N2_ref.acc'
# sample = 'Si_293K.acc'

# filepath = r'C:\Users\Usuario\Desktop\MOF\Coronado\Me-THT\22apr2024\30apr'
# reference = 'air_ref.acc'
# sample = 'sio2_ref.acc'

sample = samples[0]
reference = references[0]

thickness = 2e-4
eps_inf = 1 # glass 3.42, Si 11.6
ns = 1.9 #substate n
Z0 = cs.physical_constants['characteristic impedance of vacuum'][0]

ref_t = di.dataimport(filepath,reference)
sam_t = di.dataimport(filepath,sample) 


# %% Analysis

# moves the pulse to the center an pad 0 at the edges
ref_centered = pad.centerpad(ref_t)
sam_centered = pad.centerpad(sam_t)

#Fourier Transform

ref = fft_err.fft_err(ref_t)
sam = fft_err.fft_err(sam_t)

refw = fft_err.fft_err(ref_centered)
samw = fft_err.fft_err(sam_centered)

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

phiref = 2*np.pi*samw['Frequency (THz)']*(tref)
phisam = 2*np.pi*samw['Frequency (THz)']*(tsam)

phidiff = 2*np.pi*samw['Frequency (THz)']*(tsam-tref)
phioffset = phi.phaseoffset(ref_centered, sam_centered) #to account for different time windows starts

phidifference = phi.phaseex(refw, samw)

#transfer function
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

# nguess = 1+((phidifference-phioffset)*cs.c)/(2*np.pi*samw['Frequency (THz)']*1e12*thickness)
# kguess = -cs.c/(2*np.pi*thickness*refw['Frequency (THz)']*1e12)*np.log(((nguess+ns)**2/(1+ns)**2/nguess)*(samw['Amplitude']/refw['Amplitude']))

# #calculates complex permittivity
# eps1 = nguess**2-kguess**2
# eps2 = 2*nguess*kguess

# #loss tangent
# losstg =eps2/eps1

# #calculates complex conductivity
# realc = 4*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*nguess*kguess
# imagc = 2*np.pi*refw['Frequency (THz)']*1e12*cs.epsilon_0*(eps_inf-nguess**2+kguess**2)

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
breakpoint()
# plt.plot(refw['Frequency (THz)'],phidiff, label='$\omega (t_{sam}-t_{ref})$', linestyle='dotted')

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