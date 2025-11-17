'''Module for loading acc files. Returns THzData class object.
acc files are text documents containing the full collected data, with all scans unaveraged. The headers are indicated by leading % symbols, and each spectrum/collection is demarcated by %%.'''

import csv

class ACCLoader:

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def load(self):
        '''Loads the acc file located at self.filepath.
        Returns a THzData object containing the data and headers.'''
        headers = {}
        data = []
        current_scan = []

        with open(self.filepath, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            breakpoint()
            for row in reader:
                if not row:
                    continue
                if row[0].startswith('%%'):
                    if current_scan:
                        data.append(current_scan)
                        current_scan = []
                elif row[0].startswith('%'):
                    key_value = row[0][1:].split(':', 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        headers[key.strip()] = value.strip()
                else:
                    try:
                        numeric_row = [float(value) for value in row]
                        current_scan.append(numeric_row)
                    except ValueError:
                        continue
            if current_scan:
                data.append(current_scan)

        return THzData(data=data, headers=headers)