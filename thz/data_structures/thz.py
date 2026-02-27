'''

Future improvements:
1. Create a method for slicing/editing the dataset for averaging, to manually or automatically excluded data from the average.
2. reduce memory usage by storing data in more efficient formats - e.g. compile BaseTHzData objects into a single numpy array rather than storing each scan separately, use mapping to correlate data.
3. Add methods for advanced processing - waveform fitting, deconvolution, baseline correction, etc.
4. Refactor noise methods into separate module.'''

from __future__ import annotations
import numpy as np
import datetime
import pandas as pd
# from thz.padding import centerpad, centerpad_refactor
from thz.data_processing.preprocessing import preprocess_trace, edge_window, baseline_subtract, pad_to_window_range
from thz.data_structures.helpers import df_to_dict, dict_to_df
from thz.fft_err import fft_err #TODO: resolve circular imports later


from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass(frozen=True)
class TimeDomainStats:
    time: np.ndarray                 # (N_time,)
    mean: np.ndarray                 # (N_time,)
    std: np.ndarray                  # (N_time,)  scatter across repeats at each timepoint
    stderr: np.ndarray               # (N_time,)  std / sqrt(N_repeats)
    n_repeats: int                   # number of repeats used
    baseline_sigma: Optional[float]  # single-number sigma from baseline window on mean (if requested)
    snr_from_baseline: Optional[np.ndarray]  # (N_time,) |mean| / baseline_sigma (if requested)


def _as_slice(idx: Union[slice, np.ndarray, list, tuple, None], n: int) -> Union[slice, np.ndarray]:
    """
    Accept slice or index array/list. Validates bounds for slice.
    """
    if idx is None:
        return slice(None)

    if isinstance(idx, slice):
        # Basic bounds safety for slice
        start = 0 if idx.start is None else idx.start
        stop = n if idx.stop is None else idx.stop
        if start < 0 or stop < 0 or start > n or stop > n or start >= stop:
            raise ValueError(f"Invalid baseline_slice={idx} for length {n}.")
        return idx

    # Otherwise treat as array-like of indices / boolean mask
    arr = np.asarray(idx)
    if arr.dtype == bool and arr.shape != (n,):
        raise ValueError(f"Boolean baseline mask must have shape {(n,)}, got {arr.shape}.")
    return arr


def calculate_time_domain_stats(
    raw_data: np.ndarray,
    *,
    limit: Optional[int] = None,
    ddof: int = 1,
    baseline_slice: Union[slice, np.ndarray, list, tuple, None] = None,
    compute_baseline_snr: bool = False,
) -> TimeDomainStats:
    """
    raw_data expected shape: (N_time, 1 + N_repeats)
      - column 0 = time axis
      - columns 1: = repeated traces (same time grid)

    Returns per-timepoint mean/std/stderr across repeats.
    Optionally computes a single baseline sigma from a baseline time window on the MEAN trace,
    and SNR(t) = |mean(t)| / baseline_sigma.
    """

    if raw_data is None or raw_data.size == 0:
        # Return empty object with consistent fields
        empty = np.array([])
        return TimeDomainStats(
            time=empty, mean=empty, std=empty, stderr=empty,
            n_repeats=0, baseline_sigma=None, snr_from_baseline=None
        )

    if raw_data.ndim != 2 or raw_data.shape[1] < 2:
        raise ValueError(f"raw_data must be 2D with at least 2 columns (time + >=1 repeat). Got {raw_data.shape}.")

    time = raw_data[:, 0]
    traces = raw_data[:, 1:]  # (N_time, N_repeats_total)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")
        traces = traces[:, :limit]

    n_time, n_repeats = traces.shape
    if n_repeats < 1:
        raise ValueError("No repeat traces available after slicing.")

    mean = np.mean(traces, axis=1)

    # If only one repeat, ddof=1 would give NaNs; fall back safely
    ddof_eff = ddof if n_repeats > 1 else 0
    std = np.std(traces, axis=1, ddof=ddof_eff)
    stderr = std / np.sqrt(n_repeats)

    baseline_sigma = None
    snr = None

    if compute_baseline_snr:
        # baseline window is applied to the MEAN trace (classic approach)
        bidx = _as_slice(baseline_slice, n_time)

        # Again: if baseline region is too small, ddof=1 could be problematic
        baseline_vals = mean[bidx]
        if baseline_vals.size < 2:
            raise ValueError("Baseline region must include at least 2 points to estimate sigma.")

        baseline_sigma = float(np.std(baseline_vals, ddof=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(baseline_sigma > 0, np.abs(mean) / baseline_sigma, 0.0)

    return TimeDomainStats(
        time=time,
        mean=mean,
        std=std,
        stderr=stderr,
        n_repeats=n_repeats,
        baseline_sigma=baseline_sigma,
        snr_from_baseline=snr,
    )



class BaseTHzData:
    '''Base class for THz data structures. Holds one scan and metadata information.
    '''
    
    def __init__(self, data: np.array, headers: dict) -> None:
        self.raw_data = data  # Numpy array of [time, amplitude] pairs
        self.headers = headers  # list of header strings
        self.filename, self.scan_index = self._resolve_filename()
        self.timestamp = self._resolve_timestamp()

    def __repr__(self):
        return f"<BaseTHzData:{self.filename}, scan_{self.scan_index}, timestamp:{self.timestamp}>"
    
    def _compress_data(self):
        '''Dump the redundant X-axis if needed to save memory, deletes headers.'''
        self.raw_data = self.raw_data[:, 1]
        self.headers = None

    def _resolve_filename(self) -> tuple:
        '''Extracts filename and scan index from headers if available.'''
        for item in self.headers:
            if 'title' in item.lower():
                stringlist = item.split(' ') # Assumes format 'title filename ...'
                title = stringlist[1]
                scan_index = int(stringlist[4]) if len(stringlist) > 4 else None
                return title, scan_index
            else:
                return 'unknown_file', None
        return 'unknown_file', None # if header is empty

    def _resolve_timestamp(self) -> str:
        '''Extracts timestamp from headers if available.'''
        for item in self.headers:
            if 'date' in item.lower() and 'time' in item.lower():
                stringlist = item.split(',') # Assumes format 'Data and time,YYYY-MM-DD HH:MM:SS'
                timeobj = stringlist[1].split('.')[0].strip()
                # convert to datetime object
                timeobj = datetime.datetime.strptime(timeobj, '%Y-%m-%d %H:%M:%S')
                return timeobj
        return 'unknown_timestamp'


class THzData:
    '''Data class for holding the THz data from experiments.
    Contains multiple BaseTHzData objects for each scan, and methods for averaging and processing the data.
    
    Old dataframe compatibility: allows access via thzdata['Mean'], thzdata['Time (ps)'], etc.

    Currently implements a storage-bomb strategy where each item holds the processed data, including time-trace, fourier transformed spectrum, and referenced data. 
    Future versions will implement a more memory-efficient storage strategy with rewind features.

    '''

    # df compatibility mapping
    _column_map = {
        "Time (ps)": 0,
        "Mean": 1,
        "std error": 2,
    }

    def __init__(self, data: list, header: list, **kwargs) -> None:
        self.data_list = data  # list of BaseTHz objects for each scan
        self.raw_data = self._compile_data_array()  # np.array of compiled data from all scans
        self.headers = header if header is not None else self._grabone().headers  # retain headers from first scan # Dictionary of header information
        self.data_type = kwargs.get('data_type', None) # e.g. 'acc', 'dat', etc.
        self.filename = kwargs.get('filename', 'unknown_file')
        self.reference_filename = None

        self._meta_data = {} # stores statistical data like noise estimates, phase offset, etc. to be recalled in future processing steps
        self.processing_dict = {}  # stores processed data at various steps for rewind capability and inspection

        self._time_data = self._average_data()  # Averaged dataset
        self._time_data_headers = ["Time (ps)", "Mean", "std error"]
        self._freq_data = None
        self._freq_data_headers = None

        # self._identify_time_constant()

        self._data = self._time_data  # Current working data (time or frequency domain)

    def __getitem__(self, key):
        """
        Backwards-compatible dictionary-style access.
        Allows: thz['Mean'], thz['Time (ps)'], etc.
        """
        if key not in self._column_map:
            raise KeyError(f"{key} not found in THzData columns {list(self._column_map)}")

        col_idx = self._column_map[key]
        return self._data[:, col_idx]

    # df compatibility assignment
    def __setitem__(self, key, value):
        """
        Optional: allow assignment like thz['Mean'] = new_array.
        """
        if key not in self._column_map:
            raise KeyError(f"{key} not found in THzData columns {list(self._column_map)}")
        col_idx = self._column_map[key]
        self._data[:, col_idx] = value

    # @property
    # def reference_data(self):
    #     '''Consults the grouping service to get the reference THzData object if available. Returns the THzData object or None.'''

    @property
    def type(self) -> str:
        '''Returns the data type (e.g. 'acc', 'dat', etc.) if available.'''
        return self.__class__, self.data_type

    @property
    def time_const(self) -> float | None:
        '''Returns the time constant metadata if available. Tries to parse if not, else None.'''
        time_const = self._meta_data.get('time_constant', None)
        if time_const is None:
            time_const = self._identify_time_constant()

        return time_const

    @property
    def data(self) -> np.array:
        '''Returns the current working data array (time or frequency domain).'''
        return self._data
    # df compatibility properties
    @property
    def columns(self):
        """Backwards-compatible .columns attribute, like a DataFrame."""
        return list(self._column_map.keys())

    # df compatibility contains
    def __contains__(self, key):
        return key in self._column_map

    def __repr__(self):
        return f"\nTHzData:{self.filename}\n   -> Scans: {len(self.data_list)}\n   -> Data type: {self.data_type}\n" 
    
    def find_time_zero(self) -> tuple:
        '''Finds the index and time value of the main pulse peak in the averaged time-domain data.'''
        if self._time_data is None:
            return None, None
        mean = self._time_data[:, 1]
        time = self._time_data[:, 0]
        peak_index = int(np.argmax(np.abs(mean)))
        time_zero = time[peak_index]
        return peak_index, time_zero
    
    def _identify_time_constant(self, time_unit='ms') -> None:
        '''Work around function to pull time constant from filename, if available.'''
        if 'time' in self.filename.lower() and 'const' in self.filename.lower():
            stritem = self.filename.split(time_unit)[0].strip()
            # check string before time for float until non-float character
            stritem = stritem[::-1] # reverse string
            timeconst_str = ''
            for char in stritem:
                if char.isdigit() or char == '.':
                    timeconst_str = char + timeconst_str # adds to front so we dont need to reverse again
                else:
                    break

            try:
                timeconst = float(timeconst_str)
                self._meta_data['time_constant'] = timeconst
            except ValueError:
                print(f"Could not parse time constant from filename: {self.filename}. Setting to None.")
                self._meta_data['time_constant'] = None

        return self._meta_data.get('time_constant', None)

    def _calculate_std_error(self) -> np.array:
        '''Calculates the standard error across all scans for each time point.'''
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list))
        return std_error

    def _compile_data_array(self, limit=None) -> np.array:
        '''Takes the data from all scans and compiles it into a single numpy array.'''
        compiled_data = None
        for idx, obj in enumerate(self.data_list):
            if limit is not None and idx >= limit: # condition to limit number of scans compiled
                break
            if compiled_data is None:
                compiled_data = obj.raw_data
            else:
                compiled_data = np.column_stack((compiled_data, obj.raw_data[:, 1]))
        return compiled_data
    
    def _grabone(self, index=0) -> BaseTHzData:
        '''Returns a single BaseTHzData object from the data_list by index.'''
        return self.data_list[index]
    
    def _compress_dataset(self) -> None:
        '''Compresses the dataset by removing redundant X-axis data from each scan.'''
        for obj in self.data_list:
            obj._compress_data()

    def average_data(self, limit=None) -> np.array:
        '''Public method to average the data across number of chosen scans.'''
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list[:limit]])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list[:limit]))
        mean_data = np.mean(data_matrix, axis=0)
        time_axis = self.data_list[0].raw_data[:, 0]
        averaged_data = np.column_stack((time_axis, mean_data, std_error))
        return averaged_data

        
    def _average_data(self) -> np.array:
        '''returns array of:
         0: time (x-axis),
         1: averaged data across all scans (y-axis),
         2: Standard error as third column.'''
        
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(len(self.data_list))
        mean_data = np.mean(data_matrix, axis=0)
        time_axis = self.data_list[0].raw_data[:, 0]
        averaged_data = np.column_stack((time_axis, mean_data, std_error))
        self.processing_dict['time_domain'] = averaged_data.copy()
        return averaged_data
    
    # def centerpad_window(self, length_factor: int = 10, baseline_points: int = 10, window_alpha: float = 0.2) -> None:
        
    #     result = centerpad_window(self._data, length_factor=length_factor, baseline_points=baseline_points, window_alpha=window_alpha)
    #     self._data = result['data']
    
    # def center_pad_window(self, length_factor: int = 10) -> None:
    #     import matplotlib.pyplot as plt
    #     # breakpoint()
    #     print('Filename:', self.filename)
    #     plt.plot(self._data[:,0], self._data[:,1], label='pre-pad')
    #     result = centerpad(self._data, length_factor=length_factor)
    #     self._data = result['data']
    #     breakpoint()
    #     plt.plot(self._data[:,0], self._data[:,1], label='post-pad')
    #     plt.legend()
    #     plt.show()
    #     return self._data
    
    def _interpolate_time_axis(self, new_limits: tuple) -> None:
        '''Interpolates the averaged data to a new common time axis defined by new_limits (min, max).'''

        min_time, max_time = new_limits
        time_axis = self._data[:, 0]
        dataY = self._data[:, 1]
        resolution = time_axis[1] - time_axis[0]

        new_time_axis = np.arange(min_time, max_time, resolution)
        dataY_interp = np.interp(new_time_axis, time_axis, dataY)
        std_error_interp = np.interp(new_time_axis, time_axis, self._data[:, 2])

        self._data = np.column_stack((new_time_axis, dataY_interp, std_error_interp))
        return self._data

    def calculate_std_dev(self, limit=None) -> np.array:
        '''Calculates the standard deviation across all scans for each time point.'''
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list[:limit]])
        std_dev = np.std(data_matrix, axis=0)
        self.std_dev = std_dev
        return std_dev
    
    def center_pulse_in_window(self) -> None:
        """
        Center the main THz pulse in the current time window by padding and
        slicing so that the maximum |mean| ends up at the central index.

        This keeps the time step the same, keeps the total length the same,
        but aligns the pulse to the middle of the array.
        """
        if self._data is None:
            return

        time = self._data[:, 0]
        mean = self._data[:, 1]
        stderr = self._data[:, 2]

        N = len(time)
        if N < 3:
            return

        # find index of main pulse
        peak_index = int(np.argmax(np.abs(mean)))
        center_index = N // 2
        shift = center_index - peak_index  # +ve: pad left, -ve: pad right

        if shift == 0:
            # already centered
            return

        # padding widths for np.pad
        if shift > 0:
            pad_width = (shift, 0)    # pad left
            slice_obj = slice(0, N)   # drop extra at end
        else:
            pad_width = (0, -shift)   # pad right
            slice_obj = slice(-shift, None)  # drop extra at start

        mean_padded = np.pad(mean, pad_width, mode="constant")[slice_obj]
        stderr_padded = np.pad(stderr, pad_width, mode="constant")[slice_obj]

        # time: extend by linear extrapolation with same dt, then slice
        dt = time[1] - time[0]
        time_padded = np.pad(
            time,
            pad_width,
            mode="linear_ramp",
            end_values=(
                time[0] - dt * shift if shift > 0 else time[0],
                time[-1] + dt * (-shift) if shift < 0 else time[-1],
            ),
        )[slice_obj]

        self._data = np.column_stack((time_padded, mean_padded, stderr_padded))


    def subtract_dc_offset(self, num_points: int = 10) -> None:
        """
        Subtract a DC offset from the averaged trace using the first `num_points`
        as baseline. Modifies self._data in-place.
        """
        if self._data is None or self._data.shape[0] < num_points:
            return

        baseline = np.mean(self._data[:num_points, 1])
        self._data[:, 1] -= baseline
        # std_error unaffected (we’re just shifting mean)

    def calculate_SNR(self, limit=None, baseline_points=10) -> np.array:
        '''#current: Calculates the signal-to-noise ratio across the time domain.'''
        if self._data is None:
            return np.array([])
        
        if limit is None:
            array = self.raw_data[:, 1:]
        else:
            array = self.raw_data[:, 1:limit+1]
        
        mean = np.mean(array, axis=1)
        std = np.std(array, axis=1)#, ddof=1)
        baseline = np.median(np.abs(std[:baseline_points]))
        peak = np.max(np.abs(mean))


        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            stderr = np.where(std != 0, std / np.sqrt(array.shape[1]), 0.0)
            snr = np.where(std != 0, np.abs(mean) / std, 0.0)

        report = {
            'mean': mean,
            'std': std,
            'snr': snr,
            'stderr': stderr,
            'baseline_std': baseline,
            'peak_amplitude': peak,

        }

        return report

    def _estimate_noise_sigma(
        self,
        threshold: float = 0.05,
        min_points: int = 20,
        baseline_fraction: float = 0.3,
        ) -> float:
        """
        Estimate baseline noise sigma from the current averaged trace.
        Legacy dataframe 

        Parameters
        ----------
        threshold : float
            Fraction of the peak amplitude below which points are considered
            "baseline" (|mean| < threshold * max|mean|).
        min_points : int
            Minimum number of baseline points required before falling back to
            a simple edge-based baseline.
        baseline_fraction : float
            Fraction of the trace (from the start) to use as a fallback
            baseline region if thresholding yields too few points.

        Returns
        -------
        noise_sigma : float
            Estimated standard deviation of noise (same units as amplitude).
        """
        if self._data is None:
            return 0.0

        mean = self._data[:, 1]
        stderr = self._data[:, 2]
        N = len(mean)
        if N < 3:
            return 0.0

        peak = np.max(np.abs(mean))

        if peak == 0:
            # No obvious signal; use global stderr
            sigma = float(np.median(np.abs(stderr)))
            return sigma if np.isfinite(sigma) else 0.0

        # First attempt: baseline where |mean| is small compared to peak
        baseline_mask = np.abs(mean) < threshold * peak
        if np.count_nonzero(baseline_mask) >= min_points:
            baseline_stderr = stderr[baseline_mask]
        else:
            # Fallback: use first baseline_fraction of the trace
            end_idx = max(int(baseline_fraction * N), 1)
            baseline_stderr = stderr[:end_idx]

        sigma = float(np.median(np.abs(baseline_stderr)))
        if not np.isfinite(sigma) or sigma == 0.0:
            # Final fallback: use std of mean in baseline region
            sigma = float(np.std(mean[:end_idx], ddof=1))
        return sigma if np.isfinite(sigma) else 0.0

    def interpolate_dt(self, target_dt: float) -> None:
        """
        Interpolates the current averaged trace to a uniform time step of
        `target_dt`. Modifies self._data in-place.
        """
        if self._data is None:
            return

        time = self._data[:, 0]
        mean = self._data[:, 1]
        stderr = self._data[:, 2]

        min_time = time[0]
        max_time = time[-1]
        new_time = np.arange(min_time, max_time, target_dt)

        new_mean = np.interp(new_time, time, mean)
        new_stderr = np.interp(new_time, time, stderr)

        self._data = np.column_stack((new_time, new_mean, new_stderr))


    def pad_time_domain(
        self,
        length_factor: float = 2.0,
        max_time: float | None = None,
        ) -> None:
        """
        Extend the time-domain trace by padding at both ends so that the
        *total time window* reaches a target span. Keeps the same time step.

        The target numeric time window is:
            target_window = length_factor * max_time

        where `max_time` should normally be chosen as the largest time span
        across the dataset (max(t_max - t_min) over all traces) to ensure that
        all padded traces share the same FFT frequency axis.

        Parameters
        ----------
        length_factor : float
            Scale factor applied to `max_time`. For example:
            - 1.0 → pad all traces to the common dataset max window.
            - 2.0 → pad all traces to twice that common window.
            If <= 1 and max_time is None, no padding is applied.
        max_time : float, optional
            Target base time window (in the same units as the time axis),
            typically the largest (t_max - t_min) across the dataset.
            If None, falls back to the current trace span and behaves like
            the old length_factor * N-based padding (but in time units).
        """
        if self._data is None:
            return

        time = self._data[:, 0]
        mean = self._data[:, 1]
        stderr = self._data[:, 2]

        N = len(time)
        if N < 3:
            return

        # Time step (assumed uniform)
        dt = time[1] - time[0]
        current_span = time[-1] - time[0]  # numeric window of current trace

        # Use the dataset-wide max_time as the base span
        base_span = max_time

        if length_factor <= 1.0:
            target_span = base_span
        else:
            target_span = length_factor * base_span

        # Compute target length in samples.
        # We want a span >= target_span, so we round up.
        # Span ≈ (N_new - 1)*dt, so:
        #   N_new ≈ target_span/dt + 1
        target_length = int(np.ceil(target_span / abs(dt))) + 1

        extra_total = max(target_length - N, 0)
        if extra_total <= 0:
            return

        left_extra = extra_total // 2
        right_extra = extra_total - left_extra

        # Build new time axis by extending with constant dt
        left_times = time[0] - dt * np.arange(left_extra, 0, -1)
        right_times = time[-1] + dt * np.arange(1, right_extra + 1)
        time_ext = np.concatenate([left_times, time, right_times])


        left_pad = np.zeros(left_extra, dtype=float)
        right_pad = np.zeros(right_extra, dtype=float)

        mean_ext = np.concatenate([left_pad, mean, right_pad])
        left_stderr = np.full(left_extra, stderr[0], dtype=float)
        right_stderr = np.full(right_extra, stderr[-1], dtype=float)

        stderr_ext = np.concatenate([left_stderr, stderr, right_stderr])

        self._data = np.column_stack((time_ext, mean_ext, stderr_ext))

        self.processing_dict['centered_padded'] = {
            'data': self._data.copy(),
            'headers': ['Time (ps)', 'Mean', 'std error'],
        }  # TODO: refactor storage strategy

    
    # def pad_time_domain(
    #     self,
    #     length_factor: int = 5,
    #     max_time: float = 50.0,
    #     use_noise: bool = False,
    #     noise_threshold: float = 0.05,
    #     baseline_fraction: float = 0.3,
    #     rng: np.random.Generator | None = None,
    #     ) -> None:
    #     """
    #     Extend the time-domain trace by padding at both ends to improve
    #     frequency resolution. Keeps the same time step.

    #     If use_noise is True, padding is filled with Gaussian noise drawn
    #     from N(0, noise_sigma^2), where noise_sigma is estimated from the
    #     baseline region of the trace. Otherwise, padding is zero.

    #     Parameters
    #     ----------
    #     length_factor : int
    #         Final length will be approximately length_factor * original_length.
    #         If <= 1, no padding is applied.
    #     use_noise : bool
    #         If True, pad with synthetic noise; if False, pad with zeros.
    #     noise_threshold : float
    #         Fraction of the peak amplitude used to define baseline region:
    #         |mean| < noise_threshold * max|mean|.
    #     baseline_fraction : float
    #         Fraction of the trace (from the start) used as a fallback baseline
    #         if peak-based selection yields too few points.
    #     rng : np.random.Generator, optional
    #         Numpy random generator for reproducible noise. If None, uses
    #         np.random.default_rng().
    #     """
    #     if self._data is None:
    #         return

    #     time = self._data[:, 0]
    #     mean = self._data[:, 1]
    #     stderr = self._data[:, 2]

    #     N = len(time)
    #     if N < 3 or length_factor <= 1:
    #         return

    #     if rng is None:
    #         rng = np.random.default_rng()

    #     dt = time[1] - time[0]
    #     desired_length = int(length_factor * N)

    #     extra_total = max(desired_length - N, 0)
    #     if extra_total == 0:
    #         return

    #     left_extra = extra_total // 2
    #     right_extra = extra_total - left_extra

    #     # Build new time axis by extending with constant dt
    #     left_times = time[0] - dt * np.arange(left_extra, 0, -1)
    #     right_times = time[-1] + dt * np.arange(1, right_extra + 1)
    #     time_ext = np.concatenate([left_times, time, right_times])

    #     # Estimate baseline noise sigma if needed
    #     if use_noise:
    #         noise_sigma = self._estimate_noise_sigma(
    #             threshold=noise_threshold,
    #             baseline_fraction=baseline_fraction,
    #         )
    #     else:
    #         noise_sigma = 0.0

    #     # Mean padding: noise or zeros
    #     if use_noise and noise_sigma > 0.0:
    #         left_pad = rng.normal(loc=0.0, scale=noise_sigma, size=left_extra)
    #         right_pad = rng.normal(loc=0.0, scale=noise_sigma, size=right_extra)
    #     else:
    #         left_pad = np.zeros(left_extra, dtype=float)
    #         right_pad = np.zeros(right_extra, dtype=float)

    #     mean_ext = np.concatenate([left_pad, mean, right_pad])

    #     # Std error in padding: set to baseline noise sigma (or edge value)
    #     if use_noise and noise_sigma > 0.0:
    #         left_stderr = np.full(left_extra, noise_sigma, dtype=float)
    #         right_stderr = np.full(right_extra, noise_sigma, dtype=float)
    #     else:
    #         # fall back to edge stderr if no noise
    #         left_stderr = np.full(left_extra, stderr[0], dtype=float)
    #         right_stderr = np.full(right_extra, stderr[-1], dtype=float)

    #     stderr_ext = np.concatenate([left_stderr, stderr, right_stderr])

    #     self._data = np.column_stack((time_ext, mean_ext, stderr_ext))

    #     self.processing_dict['centered_padded'] = {'data': self._data.copy(), 'headers': ['Time (ps)', 'Mean', 'std error']} # TODO: refactor storage strategy



    def plot_current(self, *, figure_obj=None, **kwargs) -> None:
        """Plot the current averaged data.

        Accepts an optional `figure_obj` (the dataset's FigureObject). If it is
        provided the method will draw onto `figure_obj.ax` and will not call
        `plt.show()`; otherwise a new figure is created and shown.
        """
        import matplotlib.pyplot as plt

        error_bars = kwargs.get('error_bars', True)
        normalise = kwargs.get('normalise', False)
        index_axis = kwargs.get('index_axis', False)

        if self._data is None:
            print("No averaged data to plot.")
            return

        # do not mutate self._data in-place; operate on a local view
        if isinstance(self._data, pd.DataFrame):
            data_view = self._data.to_numpy()
        else:
            data_view = self._data

        if index_axis:
            time = np.arange(data_view.shape[0])
        else:
            time = data_view[:, 0]

        mean_amplitude = data_view[:, 1]
        std_error = data_view[:, 2]

        if normalise:
            max_amp = np.max(np.abs(mean_amplitude))
            if max_amp != 0:
                mean_amplitude = mean_amplitude / max_amp
                std_error = std_error / max_amp

        # Acquire axis: prefer provided FigureObject, otherwise create a temporary
        show_plot = False
        if figure_obj is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
            show_plot = True
        else:
            ax = getattr(figure_obj, 'ax', None)
            if ax is None:
                # fallback to creating a new figure if the object is malformed
                fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
                show_plot = True

        line_alpha = kwargs.get('line_alpha', 1.0)
        ax.plot(time, mean_amplitude, '-', label=self.filename, alpha=line_alpha)
        if error_bars:
            ax.fill_between(
                time,
                mean_amplitude - std_error,
                mean_amplitude + std_error,
                alpha=kwargs.get('alpha', 0.3),
                color=kwargs.get('error_color', 'tab:red'),
            )

        ax.set_title(kwargs.get('title', 'Averaged THz Data'))
        ax.set_xlabel(kwargs.get('xlabel', 'Index' if index_axis else 'Time (ps)'))
        ax.set_ylabel(kwargs.get('ylabel', 'Amplitude (a.u.)'))
        ax.legend()
        ax.grid(kwargs.get('show_grid', True))

        if show_plot:
            plt.show()

    def plot_fft_current(self, **kwargs) -> None:
        '''Plots the current FFT data with error bars as a shaded region.'''
        import matplotlib.pyplot as plt

        series = kwargs.get('series', 'fft_raw')

        if self.processing_dict.get(series, None) is None:
            print("No FFT data to plot for {}.".format(series))
            return
        fft_data = self.processing_dict[series]

        fft_data = df_to_dict(fft_data)
        data = fft_data['data']
        frequency = data[:, 0]
        amplitude = data[:, 1]
        # phase = fft_data[:, 2]

        if 'figure_obj' in kwargs:
            figure_obj = kwargs.get('figure_obj')
            ax = figure_obj.ax
            show_plot = False
        else:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
            show_plot = True
        ax.plot(frequency, amplitude, '-', label=self.filename)
        ax.set_title(kwargs.get('title', f'FFT:{series}'))
        ax.set_xlabel(kwargs.get('xlabel', 'Frequency (THz)'))
        ax.set_ylabel(kwargs.get('ylabel', 'Amplitude (a.u.)'))
        ax.legend()
        ax.grid(kwargs.get('show_grid', True))

        if show_plot:
            plt.show()
    
    @property
    def time(self) -> np.array:
        '''Returns the time axis of the working dataset in data_current.'''
        if self._data is not None:
            return self._data[:, 0]
        return None
    
    @property
    def y_mean(self) -> np.array:
        '''Returns the mean amplitude of the working dataset in data_current.'''
        if self._data is not None:
            return self._data[:, 1]
        return None
    
    @property
    def y_err(self) -> np.array:
        '''Returns the standard error of the working dataset in data_current.'''
        if self._data is not None:
            return self._data[:, 2]
        return None

    def to_legacy_dataframe(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame mimicking the original Marco-style timedata:

        Columns:
        'Time (ps)':   time axis in ps
        'Mean':        averaged signal
        'std error':   standard error across scans
        """
        if self._data is None:
            raise ValueError("THzData.data is None; nothing to convert.")

        df = pd.DataFrame(
            self._data,
            columns=["Time (ps)", "Mean", "std error"],
        )
        return df
    
    def fft_raw(self):
        '''Runs the fft_err function on the current data and returns the spectrum as a numpy array.'''
        raw_data = self.processing_dict.get('time_domain', None)
        fft_result = fft_err(raw_data)
        self.processing_dict['fft_raw'] = fft_result

        return fft_result
    
    def fft_centerpad(self):
        '''Runs the fft_err function on the current centered and padded data and returns the spectrum as a numpy array.'''
        # from thz.data_processing.fft_processing import fft_err
        centered_padded_data = self.processing_dict.get('centered_padded', None)
        # centered_padded_data = dict_to_df(centered_padded_data)
        fft_result = fft_err(centered_padded_data) # returns dictionary
        self.processing_dict['fft_centered_padded'] = fft_result
        return fft_result
    
    
    def fft_edge_windowed(self):
        '''Runs the fft_err function on the current edge-windowed data and returns the spectrum as a numpy array.'''
        # from thz.data_processing.fft_processing import fft_err

        edge_windowed_data = self.processing_dict.get('edge_windowed', None)
        # edge_windowed_data = dict_to_df(edge_windowed_data)
        fft_result = fft_err(edge_windowed_data) # returns dictionary
        self.processing_dict['fft_edge_windowed'] = fft_result
        return fft_result
    
    def baseline_subtract(self, baseline_points=10, **kwargs):
        dataY_baselined = baseline_subtract(self._data, n_points=baseline_points, **kwargs)
        data = np.column_stack((self._data[:, 0], dataY_baselined, self._data[:, 2])) 
        self.processing_dict['baseline_subtracted'] = data

    def edge_window(self, alpha=0.2, **kwargs):
        data_windowed = edge_window(self._data, alpha=alpha, **kwargs)
        data = np.column_stack((data_windowed[:, :2], self._data[:, 2])) 
        self.processing_dict['edge_windowed'] = data

    def fft(self, **kwargs):
        '''Runs the full fft processing pipeline on the current data and returns the spectrum as a numpy array.'''
        from scipy.fft import rfft, rfftfreq #rfft returns only positive frequencies
        time_axis = self._data[:, 0]
        dataY = self._data[:, 1]
        data_y_error = self._data[:, 2]

        freq = rfftfreq(len(time_axis), time_axis[1]-time_axis[0])
        amplitude = rfft(dataY, norm='ortho')
        fft_error = rfft(data_y_error, norm='ortho')

        fft_result = np.column_stack((freq, np.abs(amplitude), np.abs(fft_error)))
        self.processing_dict['fft'] = fft_result

        return fft_result
    
    def prepare_for_fft(self, 
                        baseline_points=10, 
                        pad_length_factor=5.0, 
                        window_alpha=0.2,
                        time_window=None,
                        **kwargs
                        ):
        '''Preprocesses the current time-domain data for FFT by subtracting DC offset, centering pulse, and padding.
        Currently:
        1. Baseline subtraction using initial baseline_points.
        2. Window with tukey edge-based profile (window_alpha)
        3. Padding to full time-domain range of dataset to cheat the phase offset (WIP).'''

        # data_out = preprocess_trace(self._data, baseline_points=baseline_points, pad_length_factor=pad_length_factor, window_alpha=window_alpha, show_graph=show_graph)

        if isinstance(self._data, np.ndarray):
            self._data = pd.DataFrame(self._data, columns=self._time_data_headers)
        df_baselined = baseline_subtract(self._data, n_points=baseline_points, **kwargs)
        df_windowed = edge_window(df_baselined, alpha=window_alpha, **kwargs)
        data_out = pad_to_window_range(df_windowed, time_window, pad_length_factor=pad_length_factor, **kwargs)
        # breakpoint()
        # data_out = df_windowed

        self._data = data_out
        # self._time_data_headers = data_out['headers']
        self.processing_dict['edge_windowed'] = data_out




