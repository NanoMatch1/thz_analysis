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
from thz.data_structures.decorators import array_to_dataframe_adapter

#extrapolate phase difference to zero. (to avoid refractive index divergence)

# TO DO: consider to extrapolate each dataset and not just the difference,
# maybe helps when the two maxima are way different

@array_to_dataframe_adapter(arg_name="ref", columns=["Frequency (THz)", "Amplitude", "Δ(Amplitude)", "Phase", "Δ(Phase)"])
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
def phaseoffset_numpy(ref, sam):
    '''Defines phase offset due to different time zero positions in two datasets. Uses numpy arrays as input.'''

    breakpoint()

    #TODO: slice data out of tuple
    
    t0r = ref[0, 0]  # First time value of ref
    t0s = sam[0, 0]  # First time value of sam
    
    # Define freq axis
    dt = ref[1, 0] - ref[0, 0]  # Time step
    n = len(ref[:, 0])  # Number of time points
    freq = rfftfreq(n, dt)
    
    phioffset = 2*np.pi*freq*(t0s-t0r)
    
    return phioffset