'''

Future improvements:
1. Create a method for slicing/editing the dataset for averaging, to manually or automatically excluded data from the average.
2. reduce memory usage by storing data in more efficient formats - e.g. compile BaseTHzData objects into a single numpy array rather than storing each scan separately, use mapping to correlate data.'''

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

    def plot_current(self, **kwargs) -> None:
        '''Plots the current averaged data with error bars as a shaded region.'''
        import matplotlib.pyplot as plt

        if self.data is None:
            print("No averaged data to plot.")
            return

        time = self.data[:, 0]
        mean_amplitude = self.data[:, 1]
        std_error = self.data[:, 2]

        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        plt.plot(time, mean_amplitude, '-', label='Mean')
        plt.fill_between(time, mean_amplitude - std_error, mean_amplitude + std_error, 
                 alpha=kwargs.get('alpha', 0.3), color='tab:red', label='Std Error')
        plt.title(kwargs.get('title', 'Averaged THz Data'))
        plt.xlabel(kwargs.get('xlabel', 'Time (ps)'))
        plt.ylabel(kwargs.get('ylabel', 'Amplitude (a.u.)'))
        plt.legend()
        plt.grid(True)
        plt.show()