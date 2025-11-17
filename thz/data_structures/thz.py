'''

Future improvements:
1. Create a method for slicing/editing the dataset for averaging, to manually or automatically excluded data from the average.'''

class THzData:
    '''Data class for holding the THz data from experiments.
    Attributes include: a data array of each scan,  an averaged dataset, the header data.'''

    def __init__(self, data: list, headers: dict) -> None:
        self.data = data  # List of scans, each scan is a list of [time, amplitude] pairs
        self.headers = headers  # Dictionary of header information
        self.averaged_data = self._average_data()  # Averaged dataset
        self.reference_data = None
        self.data_type = None  # 'sample' or 'reference'

    def _average_data(self) -> list:
        pass
        return []  # Placeholder for averaged data calculation