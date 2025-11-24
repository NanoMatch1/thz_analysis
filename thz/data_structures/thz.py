'''

Future improvements:
1. Create a method for slicing/editing the dataset for averaging, to manually or automatically excluded data from the average.
2. reduce memory usage by storing data in more efficient formats - e.g. compile BaseTHzData objects into a single numpy array rather than storing each scan separately, use mapping to correlate data.
3. Add methods for advanced processing - waveform fitting, deconvolution, baseline correction, etc.
4. Refactor noise methods into separate module.'''

import numpy as np
import datetime

class BaseTHzData:
    '''Base class for THz data structures. Holds one scan and metadata information.'''
    
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

    def _resolve_timestamp(self) -> str:
        '''Extracts timestamp from headers if available.'''
        for item in self.headers:
            if 'date' and 'time' in item.lower():
                stringlist = item.split(',') # Assumes format 'Data and time,YYYY-MM-DD HH:MM:SS'
                timeobj = stringlist[1].split('.')[0].strip()
                # convert to datetime object
                timeobj = datetime.datetime.strptime(timeobj, '%Y-%m-%d %H:%M:%S')
                return timeobj
        return 'unknown_timestamp'


class THzData:
    '''Data class for holding the THz data from experiments.
    Contains multiple BaseTHzData objects for each scan, and methods for averaging and processing the data.'''

    def __init__(self, data: list, header: list, **kwargs) -> None:
        self.data_list = data  # list of BaseTHz objects for each scan
        self.raw_data = self._compile_data_array()  # np.array of compiled data from all scans
        self.headers = header if not None else self._grabone().headers  # retain headers from first scan # Dictionary of header information
        self.reference_data = None
        self.data_type = None  # 'sample' or 'reference'
        self.number_of_scans = len(self.data_list)
        self.filename = kwargs.get('filename', 'unknown_file')
        self.data = self._average_data()  # Averaged dataset

    def __repr__(self):
        return f"\nTHzData:{self.filename}\n   -> Scans: {self.number_of_scans}\n   -> Data type: {self.data_type}\n" 
    
    def _calculate_std_error(self) -> np.array:
        '''Calculates the standard error across all scans for each time point.'''
        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(self.number_of_scans)
        return std_error

    def _compile_data_array(self) -> np.array:
        '''Takes the data from all scans and compiles it into a single numpy array.'''
        compiled_data = None
        for obj in self.data_list:
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
        
    def _average_data(self) -> np.array:
        '''returns array of:
         0: time (x-axis),
         1: averaged data across all scans (y-axis),
         2: Standard error as third column.'''

        data_matrix = np.array([obj.raw_data[:, 1] for obj in self.data_list])
        std_error = np.std(data_matrix, axis=0) / np.sqrt(self.number_of_scans)
        mean_data = np.mean(data_matrix, axis=0)
        time_axis = self.data_list[0].raw_data[:, 0]
        averaged_data = np.column_stack((time_axis, mean_data, std_error))
        return averaged_data
    
    def _interpolate_time_axis(self, new_limits: tuple) -> None:
        '''Interpolates the averaged data to a new common time axis defined by new_limits (min, max).'''

        min_time, max_time = new_limits
        time_axis = self.data[:, 0]
        dataY = self.data[:, 1]
        resolution = time_axis[1] - time_axis[0]

        new_time_axis = np.arange(min_time, max_time, resolution)
        dataY_interp = np.interp(new_time_axis, time_axis, dataY)
        std_error_interp = np.interp(new_time_axis, time_axis, self.data[:, 2])

        self.data = np.column_stack((new_time_axis, dataY_interp, std_error_interp))
        return self.data
    
    def center_pulse_in_window(self) -> None:
        """
        Center the main THz pulse in the current time window by padding and
        slicing so that the maximum |mean| ends up at the central index.

        This keeps the time step the same, keeps the total length the same,
        but aligns the pulse to the middle of the array.
        """
        if self.data is None:
            return

        time = self.data[:, 0]
        mean = self.data[:, 1]
        stderr = self.data[:, 2]

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

        self.data = np.column_stack((time_padded, mean_padded, stderr_padded))


    def subtract_dc_offset(self, num_points: int = 10) -> None:
        """
        Subtract a DC offset from the averaged trace using the first `num_points`
        as baseline. Modifies self.data in-place.
        """
        if self.data is None or self.data.shape[0] < num_points:
            return

        baseline = np.mean(self.data[:num_points, 1])
        self.data[:, 1] -= baseline
        # std_error unaffected (we’re just shifting mean)

    def _estimate_noise_sigma(
        self,
        threshold: float = 0.05,
        min_points: int = 20,
        baseline_fraction: float = 0.3,
        ) -> float:
        """
        Estimate baseline noise sigma from the current averaged trace.

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
        if self.data is None:
            return 0.0

        mean = self.data[:, 1]
        stderr = self.data[:, 2]
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
    
    def pad_time_domain(
        self,
        length_factor: int = 5,
        use_noise: bool = False,
        noise_threshold: float = 0.05,
        baseline_fraction: float = 0.3,
        rng: np.random.Generator | None = None,
        ) -> None:
        """
        Extend the time-domain trace by padding at both ends to improve
        frequency resolution. Keeps the same time step.

        If use_noise is True, padding is filled with Gaussian noise drawn
        from N(0, noise_sigma^2), where noise_sigma is estimated from the
        baseline region of the trace. Otherwise, padding is zero.

        Parameters
        ----------
        length_factor : int
            Final length will be approximately length_factor * original_length.
            If <= 1, no padding is applied.
        use_noise : bool
            If True, pad with synthetic noise; if False, pad with zeros.
        noise_threshold : float
            Fraction of the peak amplitude used to define baseline region:
            |mean| < noise_threshold * max|mean|.
        baseline_fraction : float
            Fraction of the trace (from the start) used as a fallback baseline
            if peak-based selection yields too few points.
        rng : np.random.Generator, optional
            Numpy random generator for reproducible noise. If None, uses
            np.random.default_rng().
        """
        if self.data is None:
            return

        time = self.data[:, 0]
        mean = self.data[:, 1]
        stderr = self.data[:, 2]

        N = len(time)
        if N < 3 or length_factor <= 1:
            return

        if rng is None:
            rng = np.random.default_rng()

        dt = time[1] - time[0]
        desired_length = int(length_factor * N)

        extra_total = max(desired_length - N, 0)
        if extra_total == 0:
            return

        left_extra = extra_total // 2
        right_extra = extra_total - left_extra

        # Build new time axis by extending with constant dt
        left_times = time[0] - dt * np.arange(left_extra, 0, -1)
        right_times = time[-1] + dt * np.arange(1, right_extra + 1)
        time_ext = np.concatenate([left_times, time, right_times])

        # Estimate baseline noise sigma if needed
        if use_noise:
            noise_sigma = self._estimate_noise_sigma(
                threshold=noise_threshold,
                baseline_fraction=baseline_fraction,
            )
        else:
            noise_sigma = 0.0

        # Mean padding: noise or zeros
        if use_noise and noise_sigma > 0.0:
            left_pad = rng.normal(loc=0.0, scale=noise_sigma, size=left_extra)
            right_pad = rng.normal(loc=0.0, scale=noise_sigma, size=right_extra)
        else:
            left_pad = np.zeros(left_extra, dtype=float)
            right_pad = np.zeros(right_extra, dtype=float)

        mean_ext = np.concatenate([left_pad, mean, right_pad])

        # Std error in padding: set to baseline noise sigma (or edge value)
        if use_noise and noise_sigma > 0.0:
            left_stderr = np.full(left_extra, noise_sigma, dtype=float)
            right_stderr = np.full(right_extra, noise_sigma, dtype=float)
        else:
            # fall back to edge stderr if no noise
            left_stderr = np.full(left_extra, stderr[0], dtype=float)
            right_stderr = np.full(right_extra, stderr[-1], dtype=float)

        stderr_ext = np.concatenate([left_stderr, stderr, right_stderr])

        self.data = np.column_stack((time_ext, mean_ext, stderr_ext))


    def plot_current(self, **kwargs) -> None:
        '''Plots the current averaged data with error bars as a shaded region.'''
        import matplotlib.pyplot as plt

        if self.data is None:
            print("No averaged data to plot.")
            return

        time = self.data[:, 0]
        mean_amplitude = self.data[:, 1]
        std_error = self.data[:, 2]

        if 'figure_obj' in kwargs:
            figure_obj = kwargs.get('figure_obj')
            ax = figure_obj.ax
            show_plot = False
        else:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
            show_plot = True
        ax.plot(time, mean_amplitude, '-', label='Mean')
        ax.fill_between(time, mean_amplitude - std_error, mean_amplitude + std_error, 
                 alpha=kwargs.get('alpha', 0.3), color='tab:red', label='Std Error')
        ax.set_title(kwargs.get('title', 'Averaged THz Data'))
        ax.set_xlabel(kwargs.get('xlabel', 'Time (ps)'))
        ax.set_ylabel(kwargs.get('ylabel', 'Amplitude (a.u.)'))
        ax.legend()
        ax.grid(kwargs.get('show_grid', True))

        if show_plot:
            plt.show()