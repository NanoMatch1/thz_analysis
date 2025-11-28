# -*- coding: utf-8 -*-
"""
FFT with error propagation and careful phase unwrap (after Jepsen 2019)
Refactored to work with THzData.data (N x 3 numpy array).

Assumes columns:
  0: Time (ps)
  1: Mean signal
  2: Std error of signal

Returns a pandas DataFrame with:
  'Frequency (THz)', 'Amplitude', 'Δ(Amplitude)', 'Phase', 'Δ(Phase)'
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq   # rfft returns only positive frequencies
from math import e


def fft_err(thz_data): 

    #take first column of dataframe as time , 2nd as average and 3rd as error
    time = thz_data.time
    y_mean = thz_data.y_mean
    y_err =  thz_data.y_err
    
    #define freq axis
    freq = rfftfreq(len(time), time[1]-time[0])
    #cancel offset, computed on first 10 points of time trace
    y_mean = y_mean-np.average(y_mean[0:10])
    
    #fourier transform and error / normalized by sqrt(N)

    ft_y_mean = rfft(y_mean.values, norm='ortho') 
    ft_y_err = rfft(y_err.values,  norm='ortho')
    
    #variance of fft is fft of variance
    
    """
    note: if time series has been multiplied by a window, it will show ripples
    in the spectrum because you get the convolution of the frequency response 
    of the window and your signal
    """
    ft_variance = rfft(y_err.values**2,  norm='ortho')
    sr = np.sqrt(abs(ft_variance.real))
    si = np.sqrt(abs(ft_variance.imag))
    
    #Euler notation and error propagation 
    amplitude = abs(ft_y_mean)
    
    # informed phase unwrapping (see header for details)
    t0 = time[np.argmax(abs(y_mean))]       #find maximum time domain
    phase0 = 2*np.pi*t0*freq                #phase of the maximum
    ft_y_mean = ft_y_mean*e**(-1j*phase0)   #reduced phase
    phase = np.angle(ft_y_mean)
    
    phase = -np.unwrap(phase) #either this minus sign or complex conjugated fft (sign convention)
    phase = phase+phase0
    
    
    #error propagation from cartesian to polar coordinates 
    err_amplitude = np.sqrt((sr*ft_y_mean.real)**2+(si*ft_y_mean.imag)**2)/amplitude
    err_phase =     np.sqrt((si/ft_y_mean.real)**2+
                            (ft_y_mean.imag*sr/ft_y_mean.real**2)**2)/(1+(ft_y_mean.imag/ft_y_mean.real)**2)

    err_phase = err_phase*phase #to account for unwrapped phase
    
    #create dataframe with fft results
    dff = pd.DataFrame(np.column_stack((freq, amplitude, err_amplitude, phase,
                                        err_phase)))
    dff.columns = ['Frequency (THz)','Amplitude','Δ(Amplitude)','Phase',
                   'Δ(Phase)']
   
    return dff



def transfer_function(ref: pd.DataFrame, sam: pd.DataFrame, offset) -> pd.DataFrame:
    amplitude = sam["Amplitude"] / ref["Amplitude"]
    # phase = sam["Phase"] - ref["Phase"] - offset
    phase = phaseex(ref, sam) - offset

    err_amplitude = (
        1.0 / (ref["Amplitude"] ** 2)
        * (sam["Δ(Amplitude)"] * ref["Amplitude"] + ref["Δ(Amplitude)"] * sam["Amplitude"])
    )
    err_phase = sam["Δ(Phase)"] + ref["Δ(Phase)"]

    T = pd.DataFrame(
        np.column_stack((ref["Frequency (THz)"], amplitude, err_amplitude, phase, err_phase)),
        columns=["Frequency (THz)", "Amplitude", "Δ(Amplitude)", "Phase", "Δ(Phase)"],
    )
    return T
