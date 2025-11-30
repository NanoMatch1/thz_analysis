import numpy as np
import pandas as pd
from thz.data_structures.decorators import array_to_dataframe_adapter

@array_to_dataframe_adapter(arg_name="ref", columns=['Frequency (THz)','Amplitude','Δ(Amplitude)','Phase',
                   'Δ(Phase)'])
def compute_snr(ref, refw):
    '''Computes the dynamic range improvement in dB between two reference datasets,
    one with and one without a certain technique (e.g., averaging, filtering).
    
    Parameters:
    ref : pd.DataFrame OR np.ndarray
        Reference dataset without the technique applied. Must contain 'Frequency (THz)' and 'Amplitude' columns.
    refw : pd.DataFrame OR np.ndarray
        Reference dataset with the technique applied. Must contain 'Frequency (THz)' and 'Amplitude' columns.
    
    Returns:
    float
        Dynamic range improvement in decibels (dB).
    '''

    lo_threshold = 4
    up_threshold = 10
    nrange = ref.loc[ref['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
    nfloor_ref = nrange.mean()

    DR = np.max(ref['Amplitude'].values)/nfloor_ref

    nrangew = refw.loc[refw['Frequency (THz)'].between(lo_threshold, up_threshold), 'Amplitude']
    nfloor_refw = nrangew.mean()

    DRw = np.max(refw['Amplitude'].values)/nfloor_refw

    DRimpr = 10*np.log10(DRw/DR)
    return DRimpr
