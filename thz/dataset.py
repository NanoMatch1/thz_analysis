from io.acc_loader import ACCLoader
from data_structures.thz import THzData

class DataSet:

    def __init__(self, file_dir: str, **kwargs) -> None:
        self.file_dir = file_dir
        self.data_dict = {}
        self.sample_keys = kwargs.get('sample_keys', [])
        self.reference_keys = kwargs.get('reference_keys', [])

    def load_data(self, filename: str) -> THzData:
        '''Loads a specific acc file and returns a THzData object.'''
        import os

        filepath = os.path.join(self.file_dir, filename)
        loader = ACCLoader(filepath)
        thz_data = loader.load()
        return thz_data

    def load_all_data(self) -> None:
        '''Loads all acc files in the specified directory into the data_dict attribute.'''
        import os

        for filename in os.listdir(self.file_dir):
            if filename.endswith('.acc'):
                thz_data = self.load_data(filename)
                self.data_dict[filename] = thz_data

        return self.data_dict
    