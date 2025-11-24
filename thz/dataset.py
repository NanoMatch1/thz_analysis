import os
import matplotlib.pyplot as plt
from thz.io.acc_loader import ACCLoader
from thz.data_structures.thz import THzData

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


class DataSet:
    '''Class for managing a dataset of THzData objects loaded from a directory.'''

    def __init__(self, file_dir: str, **kwargs) -> None:
        self.file_dir = file_dir
        self.data_dict = {}
        self.sample_keys = kwargs.get('sample_keys', [])
        self.reference_keys = kwargs.get('reference_keys', [])
        self.figure_objects = {}

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

    def plot_current(self, key: str = None, **kwargs) -> None:
        '''Plots the current data for all THzData objects in the dataset.'''
        for name, thz_data in self.data_dict.items():
            figure_obj = self._generate_figure_object('main')
            thz_data.plot_current(figure_obj=figure_obj, **kwargs)
        
        plt.show()


    def plot_sn(self):
        '''Plots the signal-to-noise ratio across the time domain for the loaded THzData objects.'''
        pass
        