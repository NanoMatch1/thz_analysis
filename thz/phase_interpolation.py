# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:06:03 2023

https://doi.org/10.1007/s10762-019-00578-0

@author: Marco Ballabio
"""
#to do: replace gradient with fit on a small selected region

import numpy as np
from scipy.fft import rfftfreq
from scipy.stats import linregress

#extrapolate phase difference to zero. (to avoid refractive index divergence)

# TO DO: consider to extrapolate each dataset and not just the difference,
# maybe helps when the two maxima are way different

def phaseex(ref,sam):
    
    dff = sam['Phase']-ref['Phase']
    
    #defines range of interest
    low = 0.3
    up = 2
    x = ref['Frequency (THz)'].loc[ref['Frequency (THz)'].between(low,up)]
    y = dff.loc[ref['Frequency (THz)'].between(low,up)]
    #least squares linear regression of the phase in the range of interest
    lsq = linregress(x,y)
    ycross = lsq.intercept
    # extrapolates the phase from max amplitude to 0 THz using the gradient
    dff -= ycross

    return dff


#to account for different time windows starts (use padded data!)
def phaseoffset(ref,sam):
    
    t0r = ref.iloc[0].at['Time (ps)']
    t0s = sam.iloc[0].at['Time (ps)']
    
    #define freq axis
    freq = rfftfreq(len(ref['Time (ps)']), ref.iloc[1].at['Time (ps)']-ref.iloc[0].at['Time (ps)'])
    
    phioffset = 2*np.pi*freq*(t0s-t0r)
    
    return phioffset

