import os
import numpy as np
import matplotlib.pyplot as plt

from dataset_core.io import get_loader_for_extension
from dataset_core.data_structures.thz import THzData, BaseTHzData, THzDataReflection
from dataset_core.io.loaders.generic_loader import GenericTextLoader
from dataset_core.services.grouping import GroupingService
from dataset_core.services.database import DatabaseService


from pathlib import Path


class DataService:
    '''Custom dict-like object holds data and accesses it as needed. Holds master data dictionary and allows access to subsets via filename keys.'''
    
    def __init__(self, grouping_service: GroupingService | None = None):
        self._data_dict = {}
        self.grouping = grouping_service if grouping_service is not None else GroupingService()

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
    def data_dict(self) -> dict:
        '''Returns the full master data dictionary.'''
        return self._data_dict

    @property
    def info(self) -> str:
        '''Returns a summary string describing the DataService contents.'''
        lines = [f"DataService with {len(self._data_dict)} items."]
        for filename, item in self._data_dict.items():
            print(f"  - {filename}: {type(item).__name__}")
            metadata = getattr(item, 'metadata', None)
            if metadata is not None:
                lines.append(f"    - Metadata: {metadata}")
        return lines
    
    @property
    def help(self):
        '''Prints available methods and properties of the DataService.'''
        print("DataService Methods and Properties:")
        print("-" * 40)
        print("data_dict: dict - Full master data dictionary.")
        print("current_data: dict - Subset of data_dict filtered to what the current working dataset should be.")
        print("grouping: GroupingService - Service for managing file groupings and references.")
        print("references: list - List of filenames identified as references.")
        print("samples: list - List of filenames identified as samples.")
        print("is_reference(filename): bool - Check if a filename is a reference.")
        print("is_sample(filename): bool - Check if a filename is a sample.")
        print("info: str - Summary string describing the DataService contents.")
        print("add_item(filename, obj) - Add an item to the data dictionary.")
        print("remove_item(filename) - Remove an item from the data dictionary.")
        print("add_items(mapping) - Add multiple items to the data dictionary from a mapping.")
        print("current_data_dict() -> dict - Get a dict of current data items based on grouping service's current file list.")

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

    @property
    def data(self):
        return self._data_dict

    @property
    def current_data(self):
        """Returns a dict restricted to the filenames
        that the grouping service considers 'current'.
        """
        return self.current_data_dict()

    def current_data_dict(self) -> dict:
        '''Returns data dict filtered to current filenames from the grouping service.'''
        current_filenames = self.grouping.get_current_data_list()
        return {key: self._data_dict[key] for key in current_filenames if key in self._data_dict}

    def set_current_files(self, filenames):
        self.grouping.set_current_data_list(list(filenames))

    def update_filelist(self, filelist):
        self.grouping.update(filelist=filelist)

    def group_files(self, **kwargs):
        '''Simple grouping based on sample and reference keys provided during initialization.'''
        result = self.grouping.simple_grouping(**kwargs)

        for filename, filename_info in self.grouping.file_items.items():
            data_obj = self._data_dict.get(filename, None)
            if data_obj is None:
                continue
            data_obj.filename_info = filename_info
        
        return result


class FigureObject:
    """Class for managing matplotlib figure and axis objects for plotting."""

    def __init__(self, figsize=None, title: str | None = None) -> None:
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.closed = False
        if title is not None:
            self.figure.suptitle(title)
        # track when this figure is closed
        self._cid = self.figure.canvas.mpl_connect('close_event', self._on_close)

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

    """Class for defining physical constants and measurements used in THz data processing.

    Standard SI constants are loaded from scipy.constants.
    Pass experiment-specific values (thickness, ns, eps_inf, etc.) as keyword arguments.
    """

    def __init__(
        self,
        thickness: float | None = None,
        ns: float | None = None,
        eps_inf: float | None = None,
    ) -> None:
        import scipy.constants as cs

        self.speed_of_light = cs.c  # m/s
        self.hbar = cs.hbar  # J·s
        self.electron_charge = cs.e  # C
        self.permittivity_free_space = cs.epsilon_0  # F/m
        self.planck_constant = cs.h  # J·s
        Z0 = cs.physical_constants['characteristic impedance of vacuum'][0]

        self.impedance_free_space = Z0  # Ohm

        # Experiment-specific parameters
        self.thickness = thickness
        self.ns = ns
        self.eps_inf = eps_inf


class DataSet:
    '''Class for managing a dataset of THzData objects loaded from a directory.'''

    def __init__(
        self,
        file_dir: str,
        *,
        data_service: DataService | None = None,
        database_service: DatabaseService | None = None,
        sample_keys: list | None = None,
        reference_keys: list | None = None,
        seriesname: str | None = None,
        metadata: dict | None = None,
        file_metadata: dict | None = None,
    ) -> None:
        self.file_dir = file_dir
        self.data = data_service if data_service is not None else DataService()
        self.grouping = self.data.grouping  # alias for convenience
        self.sample_keys = sample_keys or []
        self.reference_keys = reference_keys or []
        self.figure_objects = {}
        self.seriesname = seriesname if seriesname is not None else Path(file_dir).name
        self.metadata = dict(metadata or {})
        self.file_metadata = dict(file_metadata or {})
        self.database_service = database_service if database_service is not None else DatabaseService()

        self.history = {}

    @property
    def help(self):
        '''Prints available methods and properties of the DataSet.'''
        print("DataSet Methods and Properties:")
        print("-" * 40)
        print("file_dir: str - Directory path where data files are located.")
        print("data: DataService - Service for managing loaded data objects.")
        print("grouping: GroupingService - Service for managing file groupings and references.")
        print("sample_keys: list - List of keywords used to identify sample files.")
        print("reference_keys: list - List of keywords used to identify reference files.")
        print("figure_objects: dict - Dictionary of FigureObjects for plotting.")
        print("seriesname: str - Name of the dataset series (default is directory name).")
        print("metadata: dict - Dictionary for storing dataset-level metadata.")
        print("file_metadata: dict - Dictionary for storing per-file metadata.")
        print("database_service: DatabaseService - Service for handling database interactions.")
        print("history: dict - Dictionary for tracking processing history and metrics.")

    @property
    def data_dict(self):
        '''Returns the full master data dictionary from the data service for convenience.'''
        return self.data.data_dict

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
            filename: np.asarray(data_object.raw_data)
            for filename, data_object in self.data.items()
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
            data_object = self._build_thzdata_from_raw_array(raw_data=raw_data, filename=filename)
            self.data.add_item(filename, data_object)
            self.file_metadata[filename] = dict(file_payload.get('file_metadata', {}) or {})

        self.data.update_filelist(filelist=list(self.data.data_dict.keys()))

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
        return {}

    @property
    def all_data(self) -> dict:
        """Returns the full master data dictionary (all loaded items)."""
        return self.data.data_dict

    @property
    def current_data(self) -> dict:
        """
        Returns a dict restricted to the filenames
        that the grouping service considers 'current'.
        """
        return self.data.current_data_dict()
    
    def group_files(self, **kwargs):
        '''Simple grouping based on sample and reference keys provided during initialization.

        Forwards all kwargs (e.g. ``keywords``, ``merge_extra``, ``delimiter``)
        to ``GroupingService.simple_grouping``.
        '''
        self.grouping.simple_grouping(**kwargs)

    def add_item(self, filename, obj):
        self.data.add_item(filename, obj)

    def remove_item(self, filename):
        self.data.remove_item(filename)

    def add_items(self, mapping):
        self.data.add_items(mapping)

    def set_current_files(self, filenames):
        self.grouping.set_current_data_list(list(filenames))

    def _generate_figure_object(self, key: str, **kwargs) -> FigureObject:
        """Get or create a FigureObject for a given key."""
        fig_obj = self.figure_objects.get(key, None)
        # If it doesn't exist or has been closed, make a new one
        if fig_obj is None or not fig_obj.is_alive():
            fig_obj = FigureObject(**kwargs)
            self.figure_objects[key] = fig_obj

        return fig_obj

    def load_any(self, path: str):
        """Load a single file, falling back to ``GenericTextLoader`` when no
        registered loader matches the extension or the registered loader fails."""
        ext = Path(path).suffix  # includes the dot
        Loader = get_loader_for_extension(ext)
        if Loader is not None:
            try:
                return Loader(path).load()
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Registered loader {Loader.__name__} failed for '{path}': {exc}. "
                    "Falling back to GenericTextLoader.",
                    UserWarning,
                    stacklevel=2,
                )
        return GenericTextLoader(path).load()

    @staticmethod
    def _coerce_to_thzdata(obj):
        """If *obj* is a ``Spectrum``, convert it to ``THzData``. Otherwise return as-is."""
        from dataset_core.data_structures.spectrum import Spectrum
        if isinstance(obj, Spectrum):
            return obj.to_thzdata()
        return obj

    def convert_to_thzdata(self, *, time_unit_scale: float = 1.0, data_type: str = 'generic') -> None:
        """Convert any ``Spectrum`` objects in the data service to ``THzData`` in-place.

        Iterates the full data dictionary and replaces each ``Spectrum`` with the
        result of ``spectrum.to_thzdata()``.  Objects that are already ``THzData``
        (or any other type) are left untouched.

        Parameters
        ----------
        time_unit_scale : float
            Passed through to ``Spectrum.to_thzdata()``.  Leave as 1.0 when the
            x-axis is already in picoseconds.
        data_type : str
            Label stored on each resulting ``THzData``.
        """
        from dataset_core.data_structures.spectrum import Spectrum
        for filename, obj in list(self.data.data_dict.items()):
            if isinstance(obj, Spectrum):
                self.data.data_dict[filename] = obj.to_thzdata(
                    time_unit_scale=time_unit_scale,
                    data_type=data_type,
                )

    def load_all_data(self, prefer_acc=True, file_lister=None, case_insensitive=False, explicit_dir=False) -> DataService:
        '''Loads all files in the specified directory into the data service.

        Reflection layout detection
        ---------------------------
        If ``file_dir`` contains both a ``first_reflection/`` and a
        ``second_reflection/`` subdirectory the method automatically loads the
        dataset in reflection mode: each second-reflection file is paired with its
        matching first-reflection counterpart and stored as a
        ``THzDataReflection`` object.  If those subdirectories do not exist the
        method falls back to loading all files in ``file_dir`` directly as plain
        ``THzData`` objects (original behaviour).

        Pass ``explicit_dir=True`` to suppress reflection-layout detection and
        load ``file_dir`` as a flat directory regardless.  Use this when you need
        to re-run ``segment_reflections`` preprocessing on raw files that live
        inside a directory that already contains the segmented output subdirs, or
        when debugging the raw acquisitions directly.

        Parameters
        ----------
        prefer_acc : bool
            If True, acc files of the same name are loaded instead of dat files
            when both are present.
        file_lister : callable or None
            Optional function(directory_path) -> list[str] of filenames.
            Defaults to os.listdir. Inject a replacement for testing.
        case_insensitive : bool
            If True, file matching is done in a case-insensitive manner.
        explicit_dir : bool
            If True, bypass reflection-layout detection and load files directly
            from ``file_dir`` as plain ``THzData`` objects even when
            ``first_reflection/`` and ``second_reflection/`` subdirs are present.
            Default is False.
        '''
        first_subdir = os.path.join(self.file_dir, 'first_reflection')
        second_subdir = os.path.join(self.file_dir, 'second_reflection')
        if not explicit_dir and os.path.isdir(first_subdir) and os.path.isdir(second_subdir):
            return self._load_reflection_layout(
                first_subdir,
                second_subdir,
                prefer_acc=prefer_acc,
                file_lister=file_lister,
                case_insensitive=case_insensitive,
            )

        if file_lister is None:
            file_lister = os.listdir

        raw_files = file_lister(self.file_dir)
        if prefer_acc:
            raw_files = [
                f for f in raw_files
                if not (f.endswith('.dat') and f[:-4] + '.acc' in raw_files)
            ]

        filelist = []
        for filename in raw_files:
            filepath = os.path.join(self.file_dir, filename)
            if os.path.isdir(filepath):
                continue  # skip folders
            data_object = self._coerce_to_thzdata(self.load_any(filepath))
            if data_object is None:
                continue  # no loader for this extension
            if case_insensitive is True:
                filename = filename.lower()
            self.data.add_item(filename, data_object)
            filelist.append(filename)

        if not filelist:
            raise FileNotFoundError(f"No valid data files found in {self.file_dir}. Please check the directory and file formats.")
        self.data.update_filelist(filelist=filelist)

        return self.data

    def _load_reflection_layout(
        self,
        first_dir: str,
        second_dir: str,
        *,
        prefer_acc: bool,
        file_lister,
        case_insensitive: bool,
    ) -> DataService:
        """Load a segmented reflection dataset from first_reflection/ + second_reflection/ subdirs.

        Each second-reflection file that has a matching first-reflection file is
        promoted to a ``THzDataReflection`` object.  Files with no match are
        loaded as plain ``THzData`` objects with a warning.
        """
        if file_lister is None:
            file_lister = os.listdir

        second_raw = file_lister(second_dir)
        if prefer_acc:
            second_raw = [
                f for f in second_raw
                if not (f.endswith('.dat') and f[:-4] + '.acc' in second_raw)
            ]

        first_raw = file_lister(first_dir)
        if prefer_acc:
            first_raw = [
                f for f in first_raw
                if not (f.endswith('.dat') and f[:-4] + '.acc' in first_raw)
            ]
        first_lookup = {f.lower(): f for f in first_raw}

        filelist = []
        for filename in second_raw:
            filepath = os.path.join(second_dir, filename)
            if os.path.isdir(filepath):
                continue
            second_obj = self._coerce_to_thzdata(self.load_any(filepath))
            if second_obj is None:
                continue

            stored_name = filename.lower() if case_insensitive else filename
            second_obj.filename = stored_name

            first_match_key = filename.lower()
            first_match = first_lookup.get(first_match_key)
            if first_match is not None:
                first_filepath = os.path.join(first_dir, first_match)
                first_obj = self._coerce_to_thzdata(self.load_any(first_filepath))
                if first_obj is not None:
                    first_obj.filename = stored_name
                    second_obj = THzDataReflection.from_thzdata(second_obj, first_obj)
                else:
                    print(f"Warning: could not load first-reflection counterpart for '{filename}'.")
            else:
                print(f"Warning: no first-reflection counterpart found for '{filename}'; loaded as plain THzData.")

            self.data.add_item(stored_name, second_obj)
            filelist.append(stored_name)

        if not filelist:
            raise FileNotFoundError(
                f"No valid data files found in {second_dir}. "
                "Check that second_reflection/ contains .acc or .dat files."
            )
        self.data.update_filelist(filelist=filelist)

        n_reflection = sum(
            1 for obj in self.data.values()
            if isinstance(obj, THzDataReflection)
        )
        print(
            f"Loaded {len(filelist)} files from reflection layout "
            f"({n_reflection} with paired first-reflection segments)."
        )
        return self.data
    
    def load_constants(self, constants: Constants) -> None:
        '''Loads physical constants into all data objects in the dataset.'''
        self.constants = constants
        for data_object in self.data.values():
            data_object.constants = constants
        
    def grabonedata(self):
        '''Returns one data object from the dataset for quick access.'''
        if self.data:
            return next(iter(self.data.values()))
        print("Data dictionary is empty. Load data first.")
        return None

    def plot_current(self, **kwargs) -> None:
        '''Plots the current data for all data objects in the dataset.'''
        for name, data_object in self.data.items():
            if kwargs.get('filenames', None) is not None:
                filenames = kwargs['filenames']
                if name not in filenames:
                    continue

            figure_obj = self._generate_figure_object('main')
            data_object.plot_current(figure_obj=figure_obj, **kwargs)
        
        plt.show()

    def assign_references(self, reference_type='substrate'):
        '''Workaround for convenience. Checks the filename_info from grouping and assigns the reference_filename attribute for each THzData object in the dataset.'''
        for filename, data_object in self.data.items():
            filename_info = data_object.filename_info
            if filename_info is not None:
                reference_filename = filename_info.substrate_reference
                if reference_filename is None and 'reference' in filename:
                    continue
                data_object.reference_filename = reference_filename

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

    def modify_acquisitions(self, page_size: int = 25, in_place: bool = False, export=False, export_dir: str | Path | None = None) -> dict[str, np.ndarray]:
        from acquisition_editor import load_file, edit_data, save_acc, save_dat
        """Launch the standalone interactive acquisition editor.

        Requires that the loaded data objects expose a `raw_data` attribute
        (e.g. THzData objects).

        Constructs a dict of filename to raw_data arrays, passes it to the editor.

        Parameters
        ----------
        page_size : int
            Items shown per page in Included/Excluded lists.
        in_place : bool
            If True, applies saved edits back onto each object's raw_data.
            If False (default), leaves data objects untouched.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of filename to edited raw_data arrays.
        """
        def _construct_by_file_dict():
            raw_dict = {
                filename: np.asarray(data_object.raw_data)
                for filename, data_object in self.data.items()
            }
            
            return raw_dict

        edited_data = {}

        for filename, raw_data in _construct_by_file_dict().items():
            by_file_dict = {
                "filename": filename,
                "header": [],
                "scan_headers": [],
                "data": raw_data,
            }
            edited = edit_data(by_file_dict, page_size=page_size)
            edited_data[filename] = edited["data"]

        if in_place:
            for filename, new_raw in edited_data.items():
                data_object = self.data.get(filename)
                if data_object is not None:
                    print(f"Updating raw_data for '{filename}' with edited array of shape {new_raw.shape}.")
                    data_object.update_data(new_raw)
                else:
                    print(f"Warning: '{filename}' not found in data service. Skipping update.")

        if export:
            if export_dir is None:
                export_dir = os.path.join(self.file_dir, "export")
            for filename, new_raw in edited_data.items():
                save_data = {"filename": filename, "data": new_raw,}
                filepath = os.path.join(export_dir, filename)
                save_acc(save_data, filepath)
                save_dat(save_data, filepath)

        return edited_data
    
    def save_database(self, seriesname=None, database_dir=r"C:\Users\Samuel\Data\database"):
        '''Saves dataset to a pickle file in the database directory. Never overwrites — appends a numeric index suffix if the filename already exists. Logs the save to a chronological index file.'''
        import pickle
        if seriesname is not None:
            self.seriesname = seriesname.strip()
        else:
            seriesname = input("Enter a name for this dataset series (used for database filename): ").strip()
            if not seriesname:
                print("No name entered. Aborting save.")
                return

        if not (os.path.exists(database_dir) and os.path.isdir(database_dir)):
            print(f"Database directory '{database_dir}' does not exist. Aborting save.")
            return

        base_name = f"{seriesname}_db"
        db_path = os.path.join(database_dir, f"{base_name}.pkl")
        counter = 1
        while os.path.exists(db_path):
            db_path = os.path.join(database_dir, f"{base_name}_{counter}.pkl")
            counter += 1

        filename = os.path.basename(db_path)

        notes = input("Enter notes/comments for this save (or press Enter to skip): ").strip()

        state = {
            "data_dict": self.data.data_dict,
            "grouping_state": self.grouping.get_state(),
            "notes": notes,
        }
        with open(db_path, "wb") as f:
            pickle.dump(state, f)

        index_path = os.path.join(database_dir, "_database_index.txt")
        with open(index_path, "a") as f:
            f.write(f"{filename}|{notes}\n")

        print(f"Saved dataset to database at {db_path}.")

    def load_database(self, seriesname=None, index=None, database_dir=r"C:\Users\Samuel\Data\database"):
        '''Loads a dataset from the database directory. Displays a chronologically ordered list (latest last), optionally filtered by a search query, and prompts the user to select one.
        
        Shortcuts:
          seriesname : str  — used as a search filter; auto-selects if only one match.
          index      : int or str — directly selects from the full ordered list by 1-based position
                       (same numbering shown in the printed list). Negative indices count from the end.
        '''
        import pickle
        if not (os.path.exists(database_dir) and os.path.isdir(database_dir)):
            print(f"Database directory '{database_dir}' does not exist.")
            return

        index_path = os.path.join(database_dir, "_database_index.txt")
        if not os.path.exists(index_path):
            print("No database index found. No datasets have been saved yet.")
            return

        with open(index_path, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        def _parse_index_line(line):
            parts = line.split("|", 1)
            return parts[0], parts[1] if len(parts) > 1 else ""

        all_entries = [_parse_index_line(line) for line in raw_lines]
        existing_entries = [
            (fname, notes) for fname, notes in all_entries
            if os.path.exists(os.path.join(database_dir, fname))
        ]
        if not existing_entries:
            print("No database files found.")
            return

        ordered_entries = list(existing_entries)

        # Direct index shortcut: bypass search and selection entirely.
        if index is not None:
            try:
                idx = int(index)
                # Convert 1-based positive index; negative indices work naturally.
                if idx > 0:
                    idx -= 1
                selected_filename = ordered_entries[idx][0]
            except (ValueError, IndexError):
                print(f"Invalid index '{index}'. Must be an integer within the list range.")
                return
            print(f"Auto-selected [{int(index)}]: {selected_filename}")
        else:
            if seriesname is not None:
                query = seriesname.strip()
            else:
                query = input("Enter a search query to filter datasets (or press Enter to show all): ").strip()

            filtered_entries = [
                (fname, notes) for fname, notes in ordered_entries
                if query.lower() in fname.lower() or query.lower() in notes.lower()
            ] if query else ordered_entries
            if not filtered_entries:
                print(f"No datasets matching '{query}'.")
                return

            if len(filtered_entries) == 1:
                selected_filename = filtered_entries[0][0]
                print(f"Auto-selected: {selected_filename}")
            else:
                print("\nAvailable datasets (latest first):")
                for i, (fname, notes) in enumerate(filtered_entries):
                    print(f"  [{i + 1}] {fname}")
                    if notes:
                        print(f"       {notes}")

                selection = input("\nEnter the number of the dataset to load: ").strip()
                try:
                    selection_index = int(selection) - 1
                    if selection_index < 0 or selection_index >= len(filtered_entries):
                        print("Invalid selection.")
                        return
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    return

                selected_filename = filtered_entries[selection_index][0]

        db_path = os.path.join(database_dir, selected_filename)

        with open(db_path, "rb") as f:
            state = pickle.load(f)

        self.data._data_dict = state.get("data_dict", {})
        grouping_state = state.get("grouping_state", None)
        if grouping_state is not None:
            self.grouping.restore_state(grouping_state)

        self.seriesname = selected_filename.replace("_db.pkl", "").replace(".pkl", "")
        print(f"Loaded dataset from {db_path}.")

        return True

    def save_state(self, savedir=None):
        import pickle
        """Saves the current state of data service and grouping serivice for faster reloading later."""
        if savedir is None:
            savedir = self.file_dir
        pickle_path = os.path.join(savedir, f"{self.seriesname}_state.pkl")
        state = {
            "data_dict": self.data.data_dict,
            "grouping_state": self.grouping.get_state(),
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Saved dataset state to {pickle_path}.")

    def load_state(self):
        import pickle
        """Loads the state of data service and grouping service from a previous save."""
        pickle_path = os.path.join(self.file_dir, f"{self.seriesname}_state.pkl")
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"No saved state found at {pickle_path}. Please save state first.")

        with open(pickle_path, "rb") as f:
            state = pickle.load(f)

        self.data._data_dict = state.get("data_dict", {})
        grouping_state = state.get("grouping_state", None)
        if grouping_state is not None:
            self.grouping.restore_state(grouping_state)

        print(f"Loaded dataset state from {pickle_path}.")
