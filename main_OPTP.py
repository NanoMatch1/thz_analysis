# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:15:15 2024

@author: Marco Ballabio
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

filepath = r'C:\Users\Usuario\Desktop\Renhao\MOF-graphene\08aug2024'
reference = '3_mof-graphene_ref.dat'
sample = '2D_3_mof-graphene_OPTP_170mW_day2_sam.dat'

# filepath = r'C:\Users\Usuario\Desktop\Renhao\MXene\28may24'
# reference = 'MXene_tds.acc'
# sample = '2d_MXene_optp_297ps_360mW.acc'

# filepath = r'C:\Users\Usuario\Desktop\Renhao\MXene\28may24'
# reference = 'mof-MXene-MOF_day2_spot2_ref.acc'
# sample = '2DOPTP800_mof-MXene-MOF_day2_spot2_sam.acc'


# filepath = r'C:\Users\Usuario\Documents\MPIP\BHT_MOF'
# reference = '2D_BHT_MOF_500nm_pk_ref.acc'
# sample = '2D_BHT_MOF_500nm_pk_sam.acc'


thickness = 1.36e-3
G0 = cs.e**2/(4*cs.hbar) # 6.08534e-5 conversion factor Siemens to G0
eps_inf = 11.8
Z0 = cs.physical_constants['characteristic impedance of vacuum'][0]
n1 = 1
n2 = 3.4

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
nrange_ref = ref.loc[ref['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
nfloor_ref = nrange_ref.mean()
nrange_sam = sam.loc[sam['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
nfloor_sam = nrange_sam.mean()

DR_ref = np.max(ref['Amplitude'].values)/nfloor_ref
DR_sam = np.max(sam['Amplitude'].values)/nfloor_sam

nrange_refw = refw.loc[refw['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
nfloor_refw = nrange_refw.mean()
nrange_samw = samw.loc[samw['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
nfloor_samw = nrange_samw.mean()

DR_refw = np.max(refw['Amplitude'].values)/nfloor_refw
DR_samw = np.max(samw['Amplitude'].values)/nfloor_samw

DR_improvement_ref = 10*np.log10(DR_refw/DR_ref) #improvement in dB
DR_improvement_sam = 10*np.log10(DR_samw/DR_sam)

# calculates initial phase offset
tref = ref_t.iloc[np.argmax(abs(ref_t['Mean'])),0]
tsam = sam_t.iloc[np.argmax(abs(sam_t['Mean'])),0]

phiref = 2*np.pi*samw['Frequency (THz)']*(tref)
phisam = 2*np.pi*samw['Frequency (THz)']*(tsam)

phidiff = 2*np.pi*samw['Frequency (THz)']*(tsam-tref)
phioffset = phi.phaseoffset(ref_centered, sam_centered) #to account for different time windows starts

phidifference = phi.phaseex(refw, samw)

#transfer function
T = fft_err.transfer_functionOPTP(refw, samw, phioffset)

photocond = 1*(T['Amplitude']*e**(1j*T['Phase']))*(n1 + n2)/(Z0*thickness)

T['Real Δσ'] = photocond.values.real #creates new column in T
T['Imaginary Δσ'] = photocond.values.imag


# %% Plots
plt.figure(0,dpi=200)
# plt.plot(sam_t['Time (ps)']-1.1,flattop(len(sam_t['Time (ps)']))*np.max(abs(sam_t['Mean'].values))*100)
plt.plot(ref_t['Time (ps)'],ref_t['Mean'],label='Reference')
plt.plot(sam_t['Time (ps)'],sam_t['Mean']*100,label='Sample x100')
plt.plot(ref_centered['Time (ps)'],ref_centered['Mean'],label='Reference windowed', linestyle='-')
plt.plot(sam_centered['Time (ps)'],sam_centered['Mean']*100,label='Sample windowed x100')
plt.xlim(120,136)
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude (V)')
plt.legend()


plt.figure(1)
# plt.plot(ref['Frequency (THz)'],ref['Amplitude'],label='Reference')
plt.plot(refw['Frequency (THz)'],refw['Amplitude'],label='Reference windowed')
# plt.plot(sam['Frequency (THz)'],sam['Amplitude'],label='Sample')
plt.plot(samw['Frequency (THz)'],samw['Amplitude'],label='Sample windowed')
plt.legend()
plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.xlim(0,10)
plt.yscale('log')

#make dB scale
plt.figure(2)
plt.plot(ref['Frequency (THz)'],10*np.log10(ref['Amplitude']/np.max(ref['Amplitude'].values)),label='Reference')
plt.plot(refw['Frequency (THz)'],10*np.log10(refw['Amplitude']/np.max(refw['Amplitude'].values)),label='Reference windowed')
plt.plot(sam['Frequency (THz)'],10*np.log10(sam['Amplitude']/np.max(ref['Amplitude'].values)), label='Sample original')
plt.plot(samw['Frequency (THz)'],10*np.log10(samw['Amplitude']/np.max(refw['Amplitude'].values)),label='Sample windowed')
plt.legend()
plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.ylabel('dB')
plt.xlim(0,10)
plt.ylim(-50,1)
# plt.yscale('log')

plt.figure(10)
plt.title('Transfer function')
plt.plot(T['Frequency (THz)'],T['Amplitude'],label='Amplitude')
plt.legend()
plt.xticks(np.arange(0,11,1))
plt.xlabel('Frequency (THz)')
plt.xlim(0,4)
plt.ylim(0,0.1)
# plt.yscale('log')


plt.figure(3)
plt.plot(T['Frequency (THz)'],T['Phase'], label='transfer $\phi$' )
# plt.plot(T['Frequency (THz)'],refw['Phase'], label='ref $\phi$' )
# plt.plot(T['Frequency (THz)'],samw['Phase'], label='sam $\phi$' )
plt.plot(refw['Frequency (THz)'],phidifference-phioffset, label='$\Delta \phi$',linestyle='dashed' )
plt.plot(refw['Frequency (THz)'],phidiff, label='$\omega (t_{sam}-t_{ref})$', linestyle='dotted')
plt.xlim(0.0,3)
# plt.ylim(-10,2000)
plt.legend()

plt.figure(5)
plt.title('Conductivity ($S/m$)')
plt.plot(refw['Frequency (THz)'],T['Real Δσ'],label='$\sigma_1$')
plt.plot(refw['Frequency (THz)'],T['Imaginary Δσ'],label='$\sigma_2$')
plt.axhline(0,linestyle = 'dashed', color ='gray', lw = '0.5')
plt.xlim(0.2,2.6)
plt.ylim(-10,10)
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
#     freqdata = pd.concat([refw,samw,T['Real Δσ'],T['Imaginary Δσ']],
#                           axis=1,keys=['Reference','Sample','Re[Δσ]','Im[Δσ]'],
#                           names=['Dataset'])
    
#     # Ask for the destination folder
#     destination_folder = path
    
#     # Specify the separator
#     separator = '\t'

#     # # Export the DataFrame to a text file in the specified destination folder
#     file_path = f"{destination_folder}\\{filename}_OPTPtimeoutput.txt"
#     timedata.to_csv(file_path, sep=separator, index=False)
#     file_path = f"{destination_folder}\\{filename}_2Doptp.txt"
#     freqdata.to_csv(file_path, sep=separator, index=False)

#     print(f"Results exported to {file_path}")
# else:
#     print("Results not saved.")
