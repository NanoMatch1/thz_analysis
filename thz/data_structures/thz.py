'''

Future improvements:
1. Create a method for slicing/editing the dataset for averaging, to manually or automatically excluded data from the average.'''

import numpy as np

class BaseTHzData:
    '''Base class for THz data structures. Holds one scan and metadata information.'''
    
    def __init__(self, data: np.array, headers: dict) -> None:
        self.raw_data = data  # Numpy array of [time, amplitude] pairs
        self.headers = headers  # list of header strings
        self.filename, self.scan_index = self._resolve_filename()
        self.timestamp = self._resolve_timestamp()

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
                datetime = stringlist[1].strip()
                return datetime
        return 'unknown_timestamp'


class THzData:
    '''Data class for holding the THz data from experiments.
    Attributes include: a data array of each scan,  an averaged dataset, the header data.'''

    def __init__(self, data: np.array, header: list) -> None:
        self.raw_data = data  # List of scans, each scan is a list of [time, amplitude] pairs
        self.headers = header  # Dictionary of header information
        self.data = self._average_data()  # Averaged dataset
        self.reference_data = None
        self.data_type = None  # 'sample' or 'reference'
        self.number_of_scans = None

    def _average_data(self) -> list:
        pass
        if self.raw_data.shape[1] > 2:
            averaged = np.mean(self.raw_data, axis=0)
        elif self.raw_data.shape[1] == 2:
            averaged = self.raw_data[:, 1]
        else:
            print("Data format not recognized for averaging.")
        return []  # Placeholder for averaged data calculation