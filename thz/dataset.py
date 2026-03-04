import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector


import thz.io.loaders   # <-- imports package, which auto-imports all loader modules
from thz.io import get_loader_for_extension
from thz.data_structures.thz import THzData, BaseTHzData
from thz.services.grouping import GroupingService
from thz.services.database import DatabaseService
from thz.services.provenance import record_provenance
from collections.abc import Mapping
from thz.data_structures.helpers import df_to_dict, smooth_trace_savgol, interpolate_data
from thz.data_processing.acquisition_editor import edit_acquisitions_interactive
from pathlib import Path
from thz.io import get_loader_for_extension
import pandas as pd


# TODO: Adding subplots to FigureObject for multi-axis plots


class DataService:
    '''Custom dict-like object holds data and accesses it as needed. Holds master data dictionary and allows access to subsets via filename keys.'''
    
    def __init__(self):
        self._data_dict = {}
        self.all = self._data_dict # alias for convenience
        self.grouping = GroupingService()

    @property
    def references(self):
        return self.grouping.references

    @property
    def samples(self):
        return self.grouping.samples
    
    def is_reference(self, filename):
        return self.grouping.is_reference(filename)

    def is_sample(self, filename):
        return self.grouping.is_sample(filename)

    @property
    def info(self):
        print(f"DataService with {len(self._data_dict)} items.")
        for filename, item in self._data_dict.items():
            print(item)

    def __repr__(self):
        return f"DataService with {len(self._data_dict)} items."

    def __getitem__(self, filename):
        return self._data_dict.get(filename, None)

    def __setitem__(self, filename, obj):
        self._data_dict[filename] = obj

    def __iter__(self):
        filelist = self.grouping.get_current_data_list()
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

    def group_simple(self, **kwargs):
        '''Simple grouping based on sample and reference keys provided during initialization.'''
        self.grouping.simple_grouping(**kwargs)

        for filename, filename_info in self.grouping.file_items.items():
            data_obj = self._data_dict.get(filename, None)
            if data_obj is None:
                continue
            data_obj.filename_info = filename_info

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

    def add_subplot(self, *args, **kwargs):
        ax = self.figure.add_subplot(*args, **kwargs)
        return ax


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
        self.grouping = self.data.grouping # alias for convenience
        self.sample_keys = kwargs.get('sample_keys', [])
        self.reference_keys = kwargs.get('reference_keys', [])
        self.figure_objects = {}
        self.seriesname = kwargs.get('seriesname', Path(file_dir).name)
        self.metadata = dict(kwargs.get('metadata', {}) or {})
        self.file_metadata = dict(kwargs.get('file_metadata', {}) or {})
        self.database_service = kwargs.get('database_service', DatabaseService())
        # self.grouping = GroupingService()

        self.history = {}

    def set_dataset_metadata_tag(self, tag_name: str, tag_value):
        self.metadata[tag_name] = tag_value

    def remove_dataset_metadata_tag(self, tag_name: str):
        self.metadata.pop(tag_name, None)

    def set_file_metadata_tags(self, filename: str, **tags):
        current_tags = self.file_metadata.get(filename, {})
        current_tags.update(tags)
        self.file_metadata[filename] = current_tags

    def remove_file_metadata_tag(self, filename: str, tag_name: str):
        current_tags = self.file_metadata.get(filename, None)
        if not isinstance(current_tags, dict):
            return
        current_tags.pop(tag_name, None)

    def export_h5(
        self,
        path: str | Path,
        *,
        dataset_metadata: dict | None = None,
        per_file_metadata: dict[str, dict] | None = None,
        compression: str | None = 'gzip',
        compression_opts: int = 1,
        store_acq_mask: bool = True,
        registry_path: str | Path | None = None,
        update_registry: bool = True,
    ) -> Path:
        """Export current dataset to one HDF5 file."""
        file_data_map = {
            filename: np.asarray(thz_data.raw_data)
            for filename, thz_data in self.data.items()
        }

        merged_dataset_metadata = dict(self.metadata)
        merged_dataset_metadata.update(dataset_metadata or {})

        merged_per_file_metadata = {
            filename: dict(tags)
            for filename, tags in self.file_metadata.items()
            if isinstance(tags, dict)
        }
        for filename, tags in (per_file_metadata or {}).items():
            merged_per_file_metadata.setdefault(filename, {})
            merged_per_file_metadata[filename].update(tags or {})

        return self.database_service.export_h5(
            path=path,
            series_name=self.seriesname,
            file_data_map=file_data_map,
            dataset_metadata=merged_dataset_metadata,
            per_file_metadata=merged_per_file_metadata,
            compression=compression,
            compression_opts=compression_opts,
            store_acq_mask=store_acq_mask,
            registry_path=registry_path,
            update_registry=update_registry,
        )

    def import_h5(
        self,
        path: str | Path,
        *,
        strict_schema: bool = False,
        registry_path: str | Path | None = None,
        update_registry: bool = True,
    ):
        """Import dataset contents from one HDF5 file into this DataSet instance."""
        loaded_payload = self.database_service.load_h5(path=path, strict_schema=strict_schema)

        self.seriesname = loaded_payload.get('series_name', self.seriesname)
        self.metadata = dict(loaded_payload.get('dataset_metadata', {}) or {})
        self.file_metadata = {}

        self.data = DataService()
        self.grouping = self.data.grouping

        for filename, file_payload in loaded_payload.get('files', {}).items():
            raw_data = np.asarray(file_payload.get('raw_data'))
            thz_data = self._build_thzdata_from_raw_array(raw_data=raw_data, filename=filename)
            self.data.add_item(filename, thz_data)
            self.file_metadata[filename] = dict(file_payload.get('file_metadata', {}) or {})

        self.data.update_filelist(filelist=list(self.data.get_all_data().keys()))

        if update_registry and self.seriesname:
            self.database_service.update_registry(
                series_name=self.seriesname,
                registry_path=registry_path,
                last_opened_path=str(Path(path).expanduser().resolve()),
            )

        return self

    def _build_thzdata_from_raw_array(self, raw_data: np.ndarray, filename: str) -> THzData:
        """Convert compiled raw array [time | acquisitions...] into THzData."""
        if raw_data.ndim != 2 or raw_data.shape[1] < 2:
            raise ValueError(
                f"Invalid raw_data shape for '{filename}': expected (N, >=2), got {raw_data.shape}."
            )

        scan_list = []
        time_axis = raw_data[:, 0]
        for acquisition_column_index in range(1, raw_data.shape[1]):
            scan_array = np.column_stack((time_axis, raw_data[:, acquisition_column_index]))
            scan_list.append(BaseTHzData(data=scan_array, headers=[]))

        return THzData(
            data=scan_list,
            header=[],
            filename=filename,
            data_type='h5',
        )

    def metadata_hook(self, filename: str, thzdata: THzData):
        """
        Hook you can implement later to pull rich metadata from your data service.

        Contract:
          - Return a JSON-serializable dict (or at least something json.dumps can handle with default=str).
          - Keep it "dynamic": arbitrary keys are fine.

        Example future implementations:
          - Look up filename in a SQLite DB / REST API
          - Pull instrument settings, operator, sample details, calibration refs, etc.
        """
        # DUMMY implementation: return empty dict by default
        return {}

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
    
    def group_simple(self, **kwargs):
        '''Simple grouping based on sample and reference keys provided during initialization.'''
        self.grouping.simple_grouping(sample_keys=self.sample_keys, reference_keys=self.reference_keys)
    
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
    
    # def load_data(self, filename: str) -> THzData:
    #     '''Loads a specific acc file and returns a THzData object.'''

    #     filepath = os.path.join(self.file_dir, filename)
    #     loader = ACCLoader(filepath)
    #     thz_data = loader.load()
    #     return thz_data
    
    # def load_dat_data(self, filename: str) -> THzData:
    #     '''Loads a specific dat file and returns a THzData object.'''
    #     #TODO: refactor to use registry

    #     filepath = os.path.join(self.file_dir, filename)
    #     loader = DATLoader(filepath)
    #     thz_data = loader.load()
    #     return thz_data
    
    # def load_all_dat_files(self) -> None:
    #     '''Loads all dat files in the specified directory into the data_dict attribute.'''
    #     #TODO: refactor to use registry

    #     filelist = []
    #     for filename in os.listdir(self.file_dir):
    #         if filename.endswith('.dat'):
    #             thz_data = self.load_dat_data(filename)
    #             self.data.add_item(filename, thz_data)
    #             filelist.append(filename)

    #     self.data.update_filelist(filelist=filelist)

        return self.data
    
    # def load_data_txt(self, filename: str) -> THzData:
    #     '''Loads a specific txt file and returns a THzData object.'''
    #     filepath = os.path.join(self.file_dir, filename)
    #     loader = TXTLoader(filepath)
    #     thz_data = loader.load()
    #     return thz_data

    def load_any(self, path: str):
        ext = Path(path).suffix  # includes the dot
        Loader = get_loader_for_extension(ext)
        return Loader(path).load()

    def load_all_data(self, prefer_acc=True) -> None:
        '''Loads all files in the specified directory into the data_dict attribute. Uses ACCLoader by default, specified by the data_type kwarg.
        prefer_acc bool specifies that if True, acc files of the same name are loaded instead of dat files, if both are present.'''

        def gen_filelist():
            files = os.listdir(self.file_dir)
            if prefer_acc:
                files = [f for f in files if not (f.endswith('.dat') and f[:-4] + '.acc' in files)] # filter out dat files if acc of same name exists
            return files

        filelist = []
        files = gen_filelist()

        for filename in files:
            if os.path.isdir(os.path.join(self.file_dir, filename)):
                continue # skip folders
            filepath = os.path.join(self.file_dir, filename)
            thz_data = self.load_any(filepath)
            self.data.add_item(filename, thz_data)
            filelist.append(filename)

        if filelist == []:
            raise FileNotFoundError(f"No valid data files found in {self.file_dir}. Please check the directory and file formats.")
        # self.grouping.update(filelist=filelist)
        self.data.update_filelist(filelist=filelist)

        return self.data
    
    def load_constants(self, constants: Constants) -> None:
        '''Loads physical constants into all THzData objects in the dataset.'''
        self.constants = constants
        for thz_data in self.data.values():
            thz_data.constants = constants
        
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

    def baseline_all(self, **kwargs):
        for filename, data_obj in self.data.items():
            data_obj.baseline_subtract(**kwargs)

    def center_pad_window_all(self, length_factor: int = 10, baseline_points: int = 10, window_alpha: float = 0.2) -> None:
        for thz_data in self.data.values():
            plt.plot(thz_data._data[:,0], thz_data._data[:,1], label='pre-process')
            thz_data.centerpad_window(length_factor=length_factor, baseline_points=baseline_points, window_alpha=window_alpha)
            plt.plot(thz_data._data[:,0], thz_data._data[:,1], label='post-process')
            plt.legend()
            plt.show()

    # @df_to_dict
    # def itentify_time_constants
    def compare_time_constants(self, normalise=True, limit=None) -> None:
        '''LEGACY: Simple comparison function. Uses normalise to decide whether to normalise std dev by time constant, for the instrument code where this parameter is not separated.'''
        for filename, thz_data in self.data.items():
            time_const = thz_data.time_const
            if time_const is None:
                continue # dont plot files without time constant info
            print(f"File: {filename}, Time Constant: {time_const} s")
            std_dev = thz_data.calculate_std_dev(limit=limit)
            if normalise:
                std_dev = std_dev * np.sqrt(time_const/2) # normalise by time constant because longer time constants integrate more shots, at the rate of 500 Hz

            plt.plot(thz_data.data[:,0], std_dev, label=f'{filename}: {time_const} s')
        plt.xlabel('Time (ps)')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation Comparison Across Time Constants: Normalised' if normalise else 'Standard Deviation Comparison Across Time Constants')
        plt.legend()
        plt.show()

    def calculate_std_dev_all(self, show_graph=False, limit=10, **kwargs) -> None:
        '''Calculates the standard deviation across all scans for each THzData object in the dataset.'''
        std_dev_dict = {}

        for filename, thz_data in self.data.items():
            std_dev = thz_data.calculate_std_dev(limit=limit)
            std_dev_dict[filename] = std_dev
        if show_graph:
            for filename, std_dev in std_dev_dict.items():
                thz_data = self.data.get(filename)
                time_axis = thz_data.data[:, 0]
                plt.plot(time_axis, std_dev, label=filename)
            plt.xlabel('Time (ps)')
            plt.ylabel('Standard Deviation')
            plt.title(f'Standard Deviation Across {limit} Scans')
            plt.legend()
            plt.show()

        return std_dev_dict

    def calculate_SNR_all(self, show_graph=False, limit=None) -> None:
        '''Calculates the signal-to-noise ratio across the time domain for all THzData objects in the dataset.'''
        snr_dict = {}

        for filename, thz_data in self.data.items():
            snr_dict[filename] = thz_data.calculate_SNR(limit=limit)

        if show_graph:
            for filename, snr in snr_dict.items():
                thz_data = self.data.get(filename)
                time_axis = thz_data.data[:, 0]
                plt.plot(time_axis, snr, label=filename)
            plt.xlabel('Time (ps)')
            plt.ylabel('Signal-to-Noise Ratio')
            plt.title('Signal-to-Noise Ratio Across Time Domain')
            plt.legend()
            plt.show()

        return snr_dict

    def plot_current(self, key: str = None, **kwargs) -> None:
        '''Plots the current data for all THzData objects in the dataset.'''
        for name, data_object in self.data.items():
            figure_obj = self._generate_figure_object('main')
            data_object.plot_current(figure_obj=figure_obj, **kwargs)
        
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
        self.data.group_simple(**kwargs)

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

    def fft_all(self):
        '''Runs FFT across different preprocessing methods for all THzData objects in the dataset.'''
        result_dict = {}
        for data_object in self.data.values():
            result = data_object.fft()
            result_dict[data_object.filename] = result
        
        return result_dict

    def transfer_function_all(self, ref_type='substrate'):
        for filename, data_object in self.data.items():
            data_object.transfer_function()


    # def interpolate_dataset(self, series_key=None):
    #     '''Interpolates all datasets in the processing_dict to a common spacing. Important'''

    def edge_window_all(self, alpha=0.2, show_graph=False, **kwargs):
        '''Applies edge windowing to all THzData objects in the dataset.'''
        for thz_data in self.data.values():
            thz_data.edge_window(alpha=alpha, show_graph=show_graph, **kwargs)

    def fft_compare(self, low_threshold = 4, up_threshold = 10, clip_data=None):
        # import thz.data_processing.phase_interpolation as phi
        # from thz.data_processing.fft_processing import transfer_function
        from thz.phase_interpolation import phaseoffset, phaseex, phaseex_v2
        from thz.fft_err import transfer_function
        from scipy import constants as cs

        '''applies analysis to compare FFT results between sample and reference datasets in the current grouping.'''

        warnings = []

        series_list = [#'fft_centered_padded', skip for debugging
             'fft_edge_windowed'
              #, 'fft_raw']
        ]
        transfer_functions = {key: {} for key in series_list}
        phase_dict = {key: {} for key in series_list}

        for key in series_list:
            for filename, thz_data in self.data.items():
                thz_reference = self.get_reference(filename, ref_type='substrate')
                if thz_reference is None:
                    continue

                time_reference_label = key.strip('fft_')
                time_sample_label = key.strip('fft_')
                ref_time = thz_reference.processing_dict[time_reference_label]
                sample_time = thz_data.processing_dict[time_sample_label]
                # # determine time delay in time domain
                t_ref = ref_time['Time (ps)'][np.argmax(abs(ref_time['Mean']))] # time at max amplitude
                t_sam = sample_time['Time (ps)'][np.argmax(abs(sample_time['Mean']))] # time at max amplitude
                delta_t_time = t_sam - t_ref
                # breakpoint()
                ref_freq = thz_reference.processing_dict[key]
                sample_freq = thz_data.processing_dict[key]
                # determine inital phase offset
                
                # phiref - send time data


                phioffset = phaseoffset(ref_time, sample_time)

                if len(phioffset) != len(ref_freq):
                    print("CRITICAL WARNING: fft_compare: PHASE DATA INVALID. To fix, ref and sam must have the same length/time axis spacing.")

                phidifference, delta_t_phase = phaseex_v2(ref_freq, sample_freq, show_graph=False)
                phidifference2, delta_t_phase2 = phaseex(ref_freq, sample_freq, show_graph=False)



                phidifference = phidifference.to_numpy()
                phidifference2 = phidifference2.to_numpy()

                delta_t_phase2 = delta_t_phase2/3 # account for phase wrangling during informed unwrapping TODO: fix properly


                if abs(delta_t_phase2) - abs(delta_t_time) > 1.0:
                    warnings.append(f"WARNING: {filename}. Large discrepancy between time-domain and frequency-domain delay measurements.")
                    warnings.append(f"Frequency-domain delay: {delta_t_phase:.3f} ps (phaseex), {delta_t_phase2:.3f} ps (phaseex_v2)")
                    warnings.append(f"Time-domain delay: {delta_t_time:.3f} ps")

                # breakpoint()
                transfer_func = transfer_function(ref_freq, sample_freq, phioffset)


                thickness = thz_data.constants.thickness
                ns = thz_data.constants.ns
                eps_inf = thz_data.constants.eps_inf

                sample_freq_axis = sample_freq['Frequency (THz)'].to_numpy() * 1e12 
                sample_amp_axis = sample_freq['Amplitude'].to_numpy()
                reference_freq_axis = ref_freq['Frequency (THz)'].to_numpy() * 1e12
                reference_amp_axis = ref_freq['Amplitude'].to_numpy()


                # breakpoint()
                # skip first point to avoid division by zero
                # calculates refractive index and extinction coefficient
                skipindex = 0 if sample_freq_axis[0] != 0 else 1
                nguess = 1+((phidifference[skipindex:]-phioffset[skipindex:])*cs.c)/(2*np.pi*sample_freq_axis[skipindex:]*thickness)

                f = sample_freq_axis[skipindex:]
                n = nguess
                As = sample_amp_axis[skipindex:]
                Ar = reference_amp_axis[skipindex:]

                log_arg = ((n+ns)**2 / ((1+ns)**2 * n)) * (As/Ar)

                print("min f:", np.nanmin(f), "Hz")
                print("nguess min/max:", np.nanmin(n), np.nanmax(n))
                if np.any(n <= 0):
                    neg = True
                else:
                    neg = False
                print("Aref min:", np.nanmin(Ar), "As min:", np.nanmin(As))
                print("log_arg finite fraction:", np.isfinite(log_arg).mean())
                print("log_arg <= 0 fraction:", (log_arg <= 0).mean())
                print("any nan in As/Ar:", np.isnan(As/Ar).any())
                print("any nan in phidiff/phioff:", np.isnan(phidifference[skipindex:]).any(), np.isnan(phioffset[skipindex:]).any())



                kguess = -cs.c/(2*np.pi*thickness*sample_freq_axis[skipindex:])*np.log(((nguess+ns)**2/(1+ns)**2/nguess)*(sample_amp_axis[skipindex:]/reference_amp_axis[skipindex:]))
                print("kguess is nan - need to debug")
                if any(np.isnan(kguess)):
                    print("kguess contains NaN values, likely due to invalid logarithm arguments.")
                    if neg:
                        warnings.append("NEG AND NAN FOUND")
                
                # breakpoint()
                #calculates complex permittivity
                eps1 = nguess**2-kguess**2
                eps2 = 2*nguess*kguess

                #loss tangent
                losstg =eps2/eps1

                #calculates complex conductivity
                realc = 4*np.pi*reference_freq_axis[1:]*cs.epsilon_0*nguess*kguess
                imagc = 2*np.pi*reference_freq_axis[1:]*cs.epsilon_0*(eps_inf-nguess**2+kguess**2)

                # breakpoint()

                physical_params = {
                    # 'Dynamic Range Improvement (dB)': DRimpr,
                    'Refractive Index (n)': nguess,
                    'Extinction Coefficient (k)': kguess,
                    'Real Permittivity ($\epsilon_{1}$)': eps1,
                    'Imaginary Permittivity ($\epsilon_{2}$)': eps2,
                    'Loss Tangent': losstg,
                    'Real Conductivity ($\sigma_{1}$)': realc,
                    'Imaginary Conductivity ($\sigma_{2}$)': imagc
                }


                transfer_functions[key][filename] = {'transfer': transfer_func, 'physical parameters': physical_params}
                phase_dict[key][filename] = {'phase difference': phidifference, 'delta t (phase)': delta_t_phase, 'delta t (phaseex_v2)': delta_t_phase2, 'delta t (time)': delta_t_time}


        if len(warnings) > 0:
            for warning in warnings:
                print(warning)
        else:
            print("No warnings detected during FFT comparison - all phase measurements are consistent.")

        return transfer_functions, phase_dict

        


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
    
    def align_on_peak(self, auto_range=None, **kwargs):
        '''Centers all THzData objects in the dataset on their main pulse peak. Operates explicitly on the time-domain data, and modifies the time-domain data in place.'''

        peak_index_dict = {}

        if auto_range is not None:
            try:
                float(auto_range[0])
                float(auto_range[1])
            
            except (ValueError, TypeError):
                print("Invalid mode tuple. Must be (float, float) representing time window around expected peak.")

        for filename, thz_data in self.data.items():
            figure_object = self._generate_figure_object('peak_alignment')
            fig, ax, state = self.span_select_extremum(thz_data._time_data, mode='abs', title='Select main pulse region to center on', figure_object=figure_object, auto_range=auto_range, **kwargs)
            peak_index_dict[filename] = state

            
        min_index = min([state['last_pick']['idx'] for state in peak_index_dict.values()])
        max_index = 0

        for filename, thz_data in self.data.items():
            state = peak_index_dict[filename]
            idx = state['last_pick']['idx']
            cut = idx - min_index
            thz_data._time_data = thz_data._time_data[cut:, :]
            max_index = max(max_index, len(thz_data._time_data))
        
        for filename, thz_data in self.data.items():
            # truncate to max length to ensure all are the same length after centering
            thz_data._time_data = thz_data._time_data[:max_index]
        

    def span_select_extremum(self,
        data,
        mode: str = "abs",          # "abs" (default), "max", or "min"
        axis_labels=("Time (ps)", "E(t) (arb.)"),
        title=None,
        marker_kwargs=None,
        on_pick=None,               # optional callback: on_pick(idx, x, y)
        figure_object=None,
        auto_range=None # set a specific index range to select the data without using the span selector, for more automated processing
    ):
        """
        Plot data[:,0] vs data[:,1] and attach a SpanSelector.
        Drag to select an x-range; returns the index + x-value of the extremum in that region.

        Parameters
        ----------
        data : array-like
            Must be a 2D array with columns [x, y].
        mode : str
            "abs": pick the point with largest |y| in the span (robust for symmetric wavepackets)
            "max": pick maximum y in the span
            "min": pick minimum y in the span
        on_pick : callable or None
            If provided, called as on_pick(idx, x, y) after selection.

        Returns
        -------
        fig, ax, state : (matplotlib Figure, Axes, dict)
            state contains last_pick = {"idx":..., "x":..., "y":...}
        """
        data = np.asarray(data)
        # data = interpolate_data(data, 0.01)
        dataX = data[:, 0]

        dataY = smooth_trace_savgol(data[:, 1], window_length=11, polyorder=3) # smooth data for more robust peak picking
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("data must be a 2D array with at least 2 columns [x, y].")


        if figure_object is None:
            fig, ax = plt.subplots()
        else:
            fig = figure_object.figure
            ax = figure_object.ax
        ax.plot(dataX, dataY)
        ax.set_xlim(dataX.min(), dataX.max())
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title or "THz trace")

        marker_kwargs = marker_kwargs or {}
        marker_kwargs.setdefault("marker", "o")
        marker_kwargs.setdefault("ms", 8)
        marker_kwargs.setdefault("mec", "k")
        marker_kwargs.setdefault("mew", 1)
        marker_kwargs.setdefault("zorder", 5)

        # A marker we move around after each selection
        pick_marker, = ax.plot([np.nan], [np.nan], **marker_kwargs)

        state = {"last_pick": None}

        def _pick_in_span(xmin, xmax):
            # Ensure xmin <= xmax
            if xmax < xmin:
                xmin, xmax = xmax, xmin

            # Find indices in span
            in_span = (dataX >= xmin) & (dataX <= xmax)
            idxs = np.flatnonzero(in_span)

            if idxs.size == 0:
                print("Span selection contains no points.")
                return

            ys = dataY[idxs]

            if mode == "abs":
                ex_index = int(np.argmax(np.abs(ys)))
            elif mode == "max":
                ex_index = int(np.argmax(ys))
            elif mode == "min":
                ex_index = int(np.argmin(ys))
            else:
                raise ValueError("mode must be one of: 'abs', 'max', 'min'")

            idx = int(idxs[ex_index])
            x0 = float(dataX[idx])
            y0 = float(dataY[idx])

            # Update marker
            pick_marker.set_data([x0], [y0])
            fig.canvas.draw_idle()

            state["last_pick"] = {"idx": idx, "x": x0, "y": y0}
            print(f"Picked {mode} extremum: idx={idx}, x={x0:.6g}, y={y0:.6g}")

            if callable(on_pick):
                on_pick(idx, x0, y0)

        span = SpanSelector(
            ax,
            onselect=_pick_in_span,
            direction="horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.2),
            # You can also set minspan to prevent tiny accidental spans
            minspan=0.0,
        )

        # Helpful instruction text
        ax.text(
            0.01, 0.99,
            "Drag to select region → picks extremum inside\n"
            f"Mode: {mode}  |  Press ESC to cancel selection",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
        )

        if auto_range is not None:
            try:
                int(auto_range[0])
                int(auto_range[1])
                xmin = dataX[auto_range[0]]
                xmax = dataX[auto_range[1]]
                print(f"Auto-selecting extremum in range: {xmin:.3f} to {xmax:.3f} (indices {auto_range[0]} to {auto_range[1]})")
                _pick_in_span(xmin, xmax)
                plt.close()
                return fig, ax, state
            except (ValueError, TypeError):
                print("Invalid auto_range. Must be a tuple of (int, int) representing the x-range indicies to automatically select.")
            

        plt.show()
        return fig, ax, state

    

    def prepare_for_fft_all(self, pad_length_factor: int = 5, baseline_points: int = 10, window_alpha:float = 0.2, **kwargs) -> None:
        '''Prepares all THzData objects for FFT by subtracting DC offset, centering pulse, and padding time-domain data.'''

        time_window = self._find_series_time_range()

        for thz_data in self.data.values():
            # thz_data.prepare_for_fft(baseline_points=baseline_points, pad_length_factor=pad_length_factor, window_alpha=window_alpha, show_graph=show_graph)
            thz_data.prepare_for_fft(time_window=time_window, baseline_points=baseline_points, pad_length_factor=pad_length_factor, window_alpha=window_alpha, **kwargs)

    def plot_all_reference_and_data(self):
        '''Plots all sample and reference THzData objects in the dataset for comparison.'''
        fig, ax = plt.subplots(2, 1, sharex=True)
        # figure_obj = self._generate_figure_object('main')
        # figure_obj.clear()
        plt.tight_layout()
        
        # ax = (figure_obj.add_subplot(1,1,1), figure_obj.add_subplot(2,1,2))

        for filename, thz_data in self.data.items():
            group_info = self.data.grouping(filename)

            if group_info.data_type == 'reference':
                data = thz_data.data
                ax[0].plot(data[:,0], data[:,1], label=f'Reference: {filename}')
                # (figure_obj=figure_obj, label=f'Reference: {filename}')
            elif group_info.data_type == 'sample':
                data = thz_data.data
                ax[1].plot(data[:,0], data[:,1], label=f'Sample: {filename}')
                # thz_data.plot_current(figure_obj=figure_obj, label=f'Sample: {filename}')
            else:
                print(f"Unknown data type for file {filename}, skipping plot.")

        # ax[0].set_title('Reference Data')
        # ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Amplitude')
        ax[0].grid()
        ax[0].legend()
        ax[0].show_xtick_labels = True
        ax[0].tick_params(axis='x', which='both', labelbottom=True)
        # reduce white space between subplots
        plt.subplots_adjust(hspace=0.3)
        # ax[1].set_title('Sample Data')
        ax[1].grid()
        ax[1].set_xlabel('Time (ps)')
        ax[1].set_ylabel('Amplitude')
        ax[1].legend()

        plt.show()

    def modify_acquisitions(self, page_size: int = 25, in_place: bool = False):
        """Launch the standalone interactive acquisition editor.

        Parameters
        ----------
        page_size : int
            Items shown per page in Included/Excluded lists.
        in_place : bool
            If True, applies saved edits back onto each `THzData.raw_data`.
            If False (default), leaves `THzData` objects untouched.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of filename to edited raw_data arrays.
        """
        raw_by_file = {
            filename: np.asarray(thz_obj.raw_data)
            for filename, thz_obj in self.data.items()
        }

        edited = edit_acquisitions_interactive(raw_by_file, page_size=page_size, in_place=False)

        if in_place:
            for filename, new_raw in edited.items():
                thz_obj = self.data.get(filename)
                if thz_obj is not None:
                    thz_obj.raw_data = np.asarray(new_raw)

        return edited