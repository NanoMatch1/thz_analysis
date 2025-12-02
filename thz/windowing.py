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

from scipy.signal.windows import  hann,hamming,flattop,boxcar,kaiser, tukey
#from scipy.signal import filtfilt
from data_structures.decorators import with_dataframe
# from data_structures.decorators import with_dataframe

from scipy.signal.windows import tukey


def tukey_window(df, alpha=0.05):
    '''Apply a Tukey window with edge tapering to the THz time-domain data. Modifies the DataFrame in place.'''


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

    return


if __name__ == "__main__":
    '''Test windowning functions'''
    import matplotlib.pyplot as plt
    w_hann = hann(100)
    w_boxcar = boxcar(100)
    w_hamming = hamming(100)
    w_flattop = flattop(100)
    w_kaiser = kaiser(100,14)
    w_tukey = tukey(100,0.2)

    plt.plot(w_hann, label='Hann')
    plt.plot(w_boxcar, label='Boxcar')
    plt.plot(w_hamming, label='Hamming')
    plt.plot(w_flattop, label='Flat-top')
    plt.plot(w_kaiser, label='Kaiser')
    plt.legend()
    plt.title('Window functions comparison')
    plt.show()

    for x in range(1,20):
        w_tukey = tukey(100, x/20)
        plt.plot(w_tukey, label=f'Tukey alpha={x/10}')
    plt.legend()
    plt.title('Tukey window functions comparison')
    plt.show()