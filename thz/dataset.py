import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thz.io.acc_loader import ACCLoader
from thz.data_structures.thz import THzData
from thz.services.grouping import GroupingService
from collections.abc import Mapping


class DataService:
    '''Custom dict-like object holds data and accesses it as needed. Holds master data dictionary and allows access to subsets via filename keys.'''
    
    def __init__(self):
        self._data_dict = {}
        self.all = self._data_dict # alias for convenience
        self.grouping = GroupingService()

    def __repr__(self):
        return f"DataService with {len(self._data_dict)} items."

    def __getitem__(self, filename):
        return self._data_dict.get(filename, None)

    def __setitem__(self, filename, obj):
        self._data_dict[filename] = obj

    def __iter__(self):
        filelist = self.grouping_service.get_current_data_list()
        data_dict = {key: self._data_dict[key] for key in filelist}
        return iter(data_dict.values())
    
    def __len__(self):
        datadict = self.grouping.get_current_data_list()
        return len(datadict)
    
    def get(self, filename):
        return self._data_dict.get(filename, None)
    
    def keys(self):
        datadict = self.grouping.get_current_data_list()
        return datadict
    
    def values(self):
        filelist = self.grouping.get_current_data_list()
        data_dict = {key: self._data_dict[key] for key in filelist}
        return data_dict.values()

    def items(self):
        filelist = self.grouping.get_current_data_list()
        data_dict = {key: self._data_dict[key] for key in filelist}
        return data_dict.items()

    def add_items(self, mapping):
        self._data_dict.update(mapping)
    
    def add_item(self, filename, obj):
        self._data_dict[filename] = obj
    
    def remove_item(self, filename):
        if filename in self._data_dict:
            del self._data_dict[filename]

    def get_data(self, filename):
        '''Returns the THzData object for a specific filename.'''
        return self._data_dict.get(filename, None)
    
    def get_data_dict(self, filenames: list):
        '''Returns a dictionary of THzData objects for the specified filenames.'''
        return {key: self._data_dict[key] for key in filenames if key in self._data_dict}

    def get_all_data(self):
        return self._data_dict
    
    def update_filelist(self, filelist):
        self.grouping.update(filelist=filelist)


class FigureObject:
    """Class for managing matplotlib figure and axis objects for plotting."""

    def __init__(self, **kwargs) -> None:
        self.figure, self.ax = plt.subplots()
        self.closed = False
        # track when this figure is closed
        self._cid = self.figure.canvas.mpl_connect('close_event', self._on_close)
        self.__dict__.update(kwargs)

    def _on_close(self, event):
        # mark as closed if this is our figure
        if event.canvas.figure is self.figure:
            self.closed = True

    def is_alive(self) -> bool:
        # double-check with matplotlib's figure registry
        return (not self.closed) and plt.fignum_exists(self.figure.number)

    def clear(self):
        self.ax.clear()

class Constants:

    """Class for defining physical constants and measurements used in THz data processing."""

    def __init__(self, **kwargs) -> None:
        import scipy.constants as cs

        self.speed_of_light = cs.c  # m/s
        self.hbar = cs.hbar  # J·s
        self.electron_charge = cs.e  # C
        self.permittivity_free_space = cs.epsilon_0  # F/m
        self.planck_constant = cs.h  # J·s
        Z0 = cs.physical_constants['characteristic impedance of vacuum'][0]

        self.impedance_free_space = Z0  # Ohm

        self.__dict__.update(kwargs)

    # def _integrity_check(self) -> None:
    #     """Check that all constants are positive numbers."""
    #     for key, value in self.__dict__.items():
    #         if not isinstance(value, (int, float)) or value <= 0:
    #             raise ValueError(f"Constant '{key}' must be a positive number.")

class DataSet:
    '''Class for managing a dataset of THzData objects loaded from a directory.'''

    def __init__(self, file_dir: str, **kwargs) -> None:
        self.file_dir = file_dir
        self.data = DataService()
        self.sample_keys = kwargs.get('sample_keys', [])
        self.reference_keys = kwargs.get('reference_keys', [])
        self.figure_objects = {}
        # self.grouping = GroupingService()

    @property
    def all_data(self) -> list:
        """Returns dictionary of all objects in the dataset."""
        return self.data.all_data()

    @property
    def data_dict(self) -> dict:
        """
        Returns a *view* over self._data_dict, restricted to the filenames
        that the grouping service considers 'current'.
        """
        current_data_list = self.data.grouping.get_current_data_list()
        current_data_dict = {key: self._data_dict[key] for key in current_data_list}

        return current_data_dict
    
    def add_item(self, filename, obj):
        self._data_dict[filename] = obj

    def remove_item(self, filename):
        if filename in self._data_dict:
            del self._data_dict[filename]

    def add_items(self, mapping):
        self._data_dict.update(mapping)

    def set_current_files(self, filenames):
        self.grouping.set_current_data_list(list(filenames))

    # @data_dict.setter
    # def data_dict(self, new_data: Mapping[str, any]) -> None:
    #     """
    #     Ingest a new set of data items.

    #     - Updates the master _data_dict with the supplied items.
    #     - Updates the grouping's 'current' list to these keys.
    #     """
    #     if not isinstance(new_data, Mapping):
    #         raise TypeError(
    #             f"data_dict must be set with a mapping of filename -> data, "
    #             f"got {type(new_data)!r}"
    #         )

    #     # 1. Update master store (merge or replace, depending on what you want)
    #     # Option A: merge into existing master
    #     for name, data in new_data.items():
    #         self._data_dict[name] = data

    #     # 2. Tell the grouping service which filenames are now 'current'
    #     if hasattr(self.grouping, "set_current_data_list"):
    #         self.grouping.set_current_data_list(list(new_data.keys()))
    #     else:
    #         raise AttributeError(
    #             "Grouping object must implement 'set_current_data_list' "
    #             "to support setting data_dict."
    #         )


    def _generate_figure_object(self, key: str, **kwargs) -> FigureObject:
        """Get or create a FigureObject for a given key."""
        fig_obj = self.figure_objects.get(key, None)
        # If it doesn't exist or has been closed, make a new one
        if fig_obj is None or not fig_obj.is_alive():
            fig_obj = FigureObject(**kwargs)
            self.figure_objects[key] = fig_obj

        return fig_obj
    
    def load_data(self, filename: str) -> THzData:
        '''Loads a specific acc file and returns a THzData object.'''
        import os

        filepath = os.path.join(self.file_dir, filename)
        loader = ACCLoader(filepath)
        thz_data = loader.load()
        return thz_data

    def load_all_data(self) -> None:
        '''Loads all acc files in the specified directory into the data_dict attribute.'''

        filelist = []

        for filename in os.listdir(self.file_dir):
            if filename.endswith('.acc'):
                thz_data = self.load_data(filename)
                self.data.add_item(filename, thz_data)
                filelist.append(filename)

        # self.grouping.update(filelist=filelist)
        self.data.update_filelist(filelist=filelist)

        return self.data
    
    def grabone(self) -> THzData:
        '''Returns one THzData object from the data_dict for quick access.'''
        if self.data:
            first_data = next(iter(self.data.values()))
            return first_data
        else:
            print("Data dictionary is empty. Load data first.")
            return None
        
    def _find_common_time_window(self):
        '''Identifies the common time window across all THzData objects.'''
        min_start = float('inf')
        max_end = float('-inf')

        for thz_data in self.data.values():
            time = thz_data.data[:, 0]
            min_start = min(min_start, min(time))
            max_end = max(max_end, max(time))

        return min_start, max_end
        
    def interpolate_pulse_window(self):
        '''Identifies the working time window across all THzData objects and interpolates them to a common shared time axis.'''
        # Determine the common time window
        min_start, max_end = self._find_common_time_window()

        for thz_data in self.data.values():
            thz_data._interpolate_time_axis(new_limits=(min_start, max_end))

    def _find_maximum_time_extent(self):
        '''Finds the maximum time extent across all THzData objects.'''
        max_extent = 0

        for thz_data in self.data.values():
            time = thz_data.data[:, 0]
            extent = max(time) - min(time)
            max_extent = max(max_extent, extent)

        return max_extent
    
    def _find_minimum_time_step(self):
        '''Finds the minimum time step across all THzData objects.'''

        min_dt = float('inf')

        for thz_data in self.data.values():
            time = thz_data.data[:, 0]
            dt = np.min(np.diff(time))
            breakpoint()
            min_dt = min(min_dt, dt)

        return min_dt

    # def prepare_all_for_fft(self, length_factor: int = 5, dc_points: int = 10) -> None:
    #     """
    #     Run standard preprocessing on all loaded THzData objects:
    #     1) subtract DC offset
    #     2) center main pulse
    #     3) pad time-domain trace
    #     """

    #     #TODO: FUndamentally broken DO NOT USE
    #     max_time = self._find_maximum_time_extent()
    #     min_dt = self._find_minimum_time_step()

    #     for thz_data in self.data.values():
    #         # breakpoint()
    #         # print(thz_data.data[:, 0])
    #         thz_data.subtract_dc_offset(num_points=dc_points)
    #         thz_data.center_pulse_in_window()
    #         thz_data.pad_time_domain(length_factor=length_factor, max_time=max_time)
    #         time_length = thz_data.data[-1, 0] - thz_data.data[0, 0]
    #         print(f'Time length after padding: {time_length} ps')

    #     # self.interpolate_pulse_window()

    def center_pad_window_all(self, length_factor: int = 10, baseline_points: int = 10, window_alpha: float = 0.2) -> None:
        for thz_data in self.data.values():
            plt.plot(thz_data._data[:,0], thz_data._data[:,1], label='pre-process')
            thz_data.centerpad_window(length_factor=length_factor, baseline_points=baseline_points, window_alpha=window_alpha)
            plt.plot(thz_data._data[:,0], thz_data._data[:,1], label='post-process')
            plt.legend()
            plt.show()





    def plot_current(self, key: str = None, **kwargs) -> None:
        '''Plots the current data for all THzData objects in the dataset.'''
        for name, thz_data in self.data.items():
            figure_obj = self._generate_figure_object('main')
            thz_data.plot_current(figure_obj=figure_obj, **kwargs)
        
        plt.show()

    def plot_fft_current(self, key: str = None, **kwargs) -> None:
        '''Plots the current FFT data for all THzData objects in the dataset.'''
        for name, thz_data in self.data.items():
            figure_obj = self._generate_figure_object('main')
            thz_data.plot_fft_current(figure_obj=figure_obj, **kwargs)
        
        plt.show()


    def plot_sn(self):
        '''Plots the signal-to-noise ratio across the time domain for the loaded THzData objects.'''
        pass


    def group_files(self, **kwargs):
        '''Groups files based on provided sample and reference keys.'''
        self.data.grouping.simple_grouping()

    def get_file_item(self, filename):
        '''Access the grouping information for a specific filename.'''
        return self.data.grouping(filename)
    
    def get_reference(self, filename, ref_type='substrate'):
        '''Consults the grouping service for the reference associated with the filename.
        ref_type returns either 'substrate' or 'air' reference.'''
        reference_filename = self.data.grouping.get_reference_filename(filename, ref_type=ref_type)
        if reference_filename is not None:
            return self.data.get(reference_filename)
        return None
    
    def centerpad_legacy(self, show_graph=False) -> None:
        '''For legacy testing, does not overwrite data, just stores centered and padded version in processing_dict.'''
        from thz.data_processing.padding import centerpad
        for thz_data in self.data.values():
            thz_data.subtract_dc_offset(num_points=10)
            # legacy function uses DataFrame input/output   
            data = pd.DataFrame(thz_data._data, columns=['Time (ps)', 'Mean', 'std error'])
            result = centerpad(data, length_factor=5)
            # thz_data._data = result['data']
            thz_data.processing_dict['centered_padded'] = result

    def fft_set(self):
        '''Runs FFT across different preprocessing methods for all THzData objects in the dataset.'''
        for thz_data in self.data.values():
            thz_data.fft_raw()
            thz_data.fft_centerpad()
            thz_data.fft_edge_windowed()

    # def interpolate_dataset(self, series_key=None):
    #     '''Interpolates all datasets in the processing_dict to a common spacing. Important'''



    def fft_compare(self, low_threshold = 4, up_threshold = 10):
        # import thz.data_processing.phase_interpolation as phi
        # from thz.data_processing.fft_processing import transfer_function
        from thz.phase_interpolation import phaseoffset, phaseex, phaseex_v2
        from thz.fft_err import transfer_function

        import numpy as np
        import pandas as pd

        '''applies analysis to compare FFT results between sample and reference datasets in the current grouping.'''

        series_list = ['fft_centered_padded', 'fft_edge_windowed', 'fft_raw']
        transfer_functions = {key: {} for key in series_list}
        phase_dict = {key: {} for key in series_list}

        for key in series_list:
            for filename, thz_data in self.data.items():
                thz_reference = self.get_reference(filename, ref_type='substrate')
                if thz_reference is None:
                    continue

                time_reference_label = key.strip('fft_')
                time_sample_label = key.strip('fft_')
                breakpoint()
                ref_time = thz_reference.processing_dict[time_reference_label]
                sample_time = thz_data.processing_dict[time_sample_label]
                # # determine time delay in time domain
                t_ref = ref_time['Time (ps)'][np.argmax(abs(ref_time['Mean']))] # time at max amplitude
                t_sam = sample_time['Time (ps)'][np.argmax(abs(sample_time['Mean']))] # time at max amplitude
                delta_t_time_ps = t_sam - t_ref
                print("Time-domain delay:", delta_t_time_ps, "ps")
                # breakpoint()
                ref_freq = thz_reference.processing_dict[key]
                sample_freq = thz_data.processing_dict[key]
                # determine inital phase offset
                
                # phiref - send time data
                phioffset = phaseoffset(ref_time, sample_time)

                breakpoint()

                phidifference, delta_t_ps = phaseex_v2(ref_freq, sample_freq, show_graph=True)
                phidifference2, delta_t_ps2 = phaseex(ref_freq, sample_freq, show_graph=True)

                transfer_func = transfer_function(ref_freq, sample_freq, phioffset)

                transfer_functions[key][filename] = transfer_func

        return transfer_functions

        


    def compare_snr(self):
        '''Compares the SNR between sample and reference datasets in the current grouping.'''
        from thz.data_processing.snr import compute_snr

        snr_results = {}

        for filename, data in self.data.items():
            group_info = self.data.grouping(filename)
            if group_info.type == 'reference':
                continue

            data_type = group_info.data_type

            if data_type == 'sample':
                sample_data = data.freq_spectrum
                reference_filename = self._find_reference_filename(filename, group_info)
                if reference_filename and reference_filename in self.data:
                    reference_data = self.data[reference_filename].freq_spectrum
                    snr_improvement = compute_snr(reference_data, sample_data)
                    snr_results[filename] = snr_improvement


        return snr_results
    
    def _find_series_time_range(self):
        '''Finds the overall time range across all THzData objects after centering.'''
        min_time = float('inf')
        max_time = float('-inf')

        for thz_data in self.data.values():
            if isinstance(thz_data._data, pd.DataFrame):
                time = thz_data._data['Time (ps)'].values
            else:
                time = thz_data.data[:, 0]
            min_time = min(min_time, min(time))
            max_time = max(max_time, max(time))

        print('Identified time window for all data:', (min_time, max_time))

        return (min_time, max_time)
    

    def prepare_for_fft_all(self, pad_length_factor: int = 5, baseline_points: int = 10, window_alpha:float = 0.2, show_graph=False) -> None:
        '''Prepares all THzData objects for FFT by subtracting DC offset, centering pulse, and padding time-domain data.'''

        time_window = self._find_series_time_range()

        for thz_data in self.data.values():
            # thz_data.prepare_for_fft(baseline_points=baseline_points, pad_length_factor=pad_length_factor, window_alpha=window_alpha, show_graph=show_graph)
            thz_data.prepare_for_fft(baseline_points=baseline_points, pad_length_factor=pad_length_factor, window_alpha=window_alpha, show_graph=show_graph, time_window=time_window)