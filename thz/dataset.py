import os
import matplotlib.pyplot as plt
from thz.io.acc_loader import ACCLoader
from thz.data_structures.thz import THzData
from thz.services.grouping import GroupingService

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
        self.data_dict = {}
        self.sample_keys = kwargs.get('sample_keys', [])
        self.reference_keys = kwargs.get('reference_keys', [])
        self.figure_objects = {}
        self.grouping = GroupingService()

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

        for filename in os.listdir(self.file_dir):
            if filename.endswith('.acc'):
                thz_data = self.load_data(filename)
                self.data_dict[filename] = thz_data

        filelist = [key for key in self.data_dict.keys()]
        self.grouping.update(filelist=filelist)

        return self.data_dict
    
    def grabone(self) -> THzData:
        '''Returns one THzData object from the data_dict for quick access.'''
        if self.data_dict:
            first_data = next(iter(self.data_dict.values()))
            return first_data
        else:
            print("Data dictionary is empty. Load data first.")
            return None
        
    def _find_common_time_window(self):
        '''Identifies the common time window across all THzData objects.'''
        min_start = float('inf')
        max_end = float('-inf')

        for thz_data in self.data_dict.values():
            time = thz_data.data[:, 0]
            min_start = min(min_start, min(time))
            max_end = max(max_end, max(time))

        return min_start, max_end
        
    def interpolate_pulse_window(self):
        '''Identifies the working time window across all THzData objects and interpolates them to a common shared time axis.'''
        # Determine the common time window
        min_start, max_end = self._find_common_time_window()

        for thz_data in self.data_dict.values():
            thz_data._interpolate_time_axis(new_limits=(min_start, max_end))


    def prepare_all_for_fft(self, length_factor: int = 5, dc_points: int = 10) -> None:
        """
        Run standard preprocessing on all loaded THzData objects:
        1) subtract DC offset
        2) center main pulse
        3) pad time-domain trace
        """
        for thz_data in self.data_dict.values():
            thz_data.subtract_dc_offset(num_points=dc_points)
            thz_data.center_pulse_in_window()
            thz_data.pad_time_domain(length_factor=length_factor)


    def plot_current(self, key: str = None, **kwargs) -> None:
        '''Plots the current data for all THzData objects in the dataset.'''
        for name, thz_data in self.data_dict.items():
            figure_obj = self._generate_figure_object('main')
            thz_data.plot_current(figure_obj=figure_obj, **kwargs)
        
        plt.show()


    def plot_sn(self):
        '''Plots the signal-to-noise ratio across the time domain for the loaded THzData objects.'''
        pass


    def group_files(self, **kwargs):
        '''Groups files based on provided sample and reference keys.'''
        file_list = [key for key in self.data_dict.keys()]
        self.grouping.simple_grouping()

    def run_fft(self):
        '''Applies FFT with error propagation to all THzData objects in the dataset.'''
        for thz_data in self.data_dict.values():
            spectrum = thz_data.run_fft()
            breakpoint()

