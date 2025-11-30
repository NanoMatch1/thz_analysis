import os
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


    def prepare_all_for_fft(self, length_factor: int = 5, dc_points: int = 10) -> None:
        """
        Run standard preprocessing on all loaded THzData objects:
        1) subtract DC offset
        2) center main pulse
        3) pad time-domain trace
        """
        for thz_data in self.data.values():
            thz_data.subtract_dc_offset(num_points=dc_points)
            thz_data.center_pulse_in_window()
            thz_data.pad_time_domain(length_factor=length_factor)


    def plot_current(self, key: str = None, **kwargs) -> None:
        '''Plots the current data for all THzData objects in the dataset.'''
        for name, thz_data in self.data.items():
            figure_obj = self._generate_figure_object('main')
            thz_data.plot_current(figure_obj=figure_obj, **kwargs)
        
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
    
    def fft_compare(self, low_threshold = 4, up_threshold = 10):
        import thz.data_processing.phase_interpolation as phi
        from thz.data_processing.fft_processing import transfer_function
        import numpy as np
        
        '''Applies FFT with error propagation to all THzData objects in the dataset.'''
        for thz_data in self.data.values():
            thz_data.fft_raw()
            thz_data.fft_centerpad()

        for filename, thz_data in self.data.items():
            thz_reference = self.get_reference(filename, ref_type='substrate')
            if thz_reference is None:
                continue

            ref_time = thz_reference.processing_dict['time_domain']
            sample_time = thz_data.processing_dict['time_domain']
            ref_centered = thz_reference.processing_dict['fft_centered_padded']
            sample_centered = thz_data.processing_dict['fft_centered_padded']
            # determine inital phase offset
            t0_ref = ref_time[np.argmax(abs(ref_time[:,1]))][0] # time at max amplitude
            t0_sam = sample_time[np.argmax(abs(sample_time[:,1]))][0] # time at max amplitude

            # phiref 
            phioffset = phi.phaseoffset_numpy(ref_centered, sample_centered)

            phidifference = phi.phaseex(ref_centered, sample_centered)

            transfer_func = transfer_function(ref_centered, sample_centered, phioffset)

            
            breakpoint()

        


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
        

