# -*- coding: utf-8 -*-
"""

Windowing of THz time-domain signals 
Expects dataframes generated in padding.py (peak of the signal at the center)

keep in mind the compromise between frequency selectivity (long window. e.g. boxcar)
and time-domain artifact avoidance (short window, e.g. flat-top) 

to do: handling more windowing functions, calling padding.py by default

Windowing seems to affect the phase of fft, more study involving analysis of 
phase difference will follow

Created on Fri Jun  2 12:11:46 2023

@author: Marco Ballabio
"""

from scipy.signal.windows import  hann,hamming,flattop,boxcar,kaiser
#from scipy.signal import filtfilt

def window(df):

    #create window function
    # w = boxcar(len(df['Time (ps)'])) # no window
    # w = hamming(len(df['Time (ps)']))
    # w = flattop(len(df['Time (ps)']))
    w = hann(len(df['Time (ps)']))
    # w = kaiser(len(df['Time (ps)']),14)
    
    #windowing
    df['Mean'] = df['Mean']*w
    df['std error'] = df['std error']*w

    return df