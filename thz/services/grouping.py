'''Service for grouping of filenames based on strings, keywords and delimiters. Used to correlate reference and sample across variable datasets suhch as temperature series.'''

from dataclasses import dataclass

@dataclass
class TemperatureItem:
    """Class for identifying a temperature from the filename.

    To construct a dataclass for filename variables, add additional fields as required. They should be boolean flags or simple types."""

    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand


    def __init__(self):
        pass


class FilenameItem:

    def __init__(self, filename: str, **kwargs):
        self.filename = filename
        self.temperature = None
        self.series = None
        self.data_type = None
        self.extra_details = None

        self.__dict__.update(kwargs)

        self._parse_filename()

    def __repr__(self):
        return f"Filename: {self.filename}\n -> type:{self.type}\n -> series:{self.series}\n -> temperature:{self.temperature}\n -> extra_details:{self.extra_details}"

    def _parse_filename(self, delimiter: str = '_', grouping: list = ['type', 'series', 'temp'], **kwargs):
        '''Parses the filename into components based on provided delimiter and grouping order.'''

        details = '.'.join(self.filename.split('.')[:-1]) # Remove file extension

        components = details.split(delimiter)
        components = [comp.strip() for comp in components if comp.strip()] # remove empty strings and whitespace

        for index, key in enumerate(grouping):
            if index < len(components):
                value = components[index]
                if key == 'type':
                    self.data_type = value
                elif key == 'series':
                    if '.' in value:
                        indicies = [i for i, ch in enumerate(value) if ch == '.']
                        value = value[:indicies[-1]]
                    self.series = value
                elif key == 'temp':
                    self.temperature = value
                else:
                    self.__dict__[key] = value

        # handle any extra components *after* assigning main fields
        if len(components) > len(grouping):
            extras = components[len(grouping):]
            if kwargs.get('merge_extra', False):
                self.series = delimiter.join([self.series, *extras])
            self.extra_details = extras
            

class GroupingService:

    '''Creates a nested dictionary structure to group filenames based on provided delimiters and keywords.'''

    def __init__(self, filelist, keywords=[''], delimiter='_', **kwargs):
        self.filelist = filelist
        self.file_items = []
        self.filename_groups = {}
        self.global_reference = {}
        self.keywords = keywords
        self.delimiter = delimiter
        self.__dict__.update(kwargs)

    def _build_fileitems(self, **kwargs):
        '''Builds FilenameItem objects for each filename in the filelist.'''
        self.file_items = []

        for filename in self.filelist:
            item = FilenameItem(filename, **kwargs)
            self.file_items.append(item)

    def simple_grouping_2(self, delimiter='_', grouping=['type', 'series', 'temp']):
        '''Groups data by slicing the filename. Expects filename to contain data outlined in grouping, and does not (yet) logically check those parameters.
        
        currently: grouping: list of strings defining the order of components in the filename. E.g. ['type', 'series', 'temp']
        '''

        self.filename_groups = {}
        self._build_fileitems(delimiter=delimiter, grouping=grouping, merge_extra=True)

        for item in self.file_items: #
            if item.data_type == 'reference' and item.series == 'substrate': # special case for substrate reference
                if 'substrate' not in self.global_reference:
                    self.global_reference['substrate'] = {item.temperature: item.filename}
                else:
                    self.global_reference['substrate'][item.temperature] = item.filename
                continue

            if item.data_type == 'reference' and 'air' in item.series.lower(): # special case for air reference
                self.global_reference['air'] = item.filename 
                continue

            if item.series not in self.filename_groups:
                self.filename_groups[item.series] = {}
            if item.temperature not in self.filename_groups[item.series]:
                self.filename_groups[item.series][item.temperature] = {}
            self.filename_groups[item.series][item.temperature][item.data_type] = item.filename
        
        # Attach global references
        for series, temps in self.filename_groups.items():
            for temp, data in temps.items():
                if 'reference' not in data.keys():
                    if 'substrate' in self.global_reference and temp in self.global_reference['substrate']:
                        data['reference'] = self.global_reference['substrate'][temp]

        print("Completed simple grouping of filenames:")
        print(self.filename_groups)
        print(self.global_reference)

    def integrity_check(self):
        '''Performs an integrity check on the grouped filenames to ensure each group has the expected components required for analysis.'''

        warnings = []

        if 'air' not in self.global_reference.keys():
            warnings.append("Warning: No air reference found in global references.")
        if 'substrate' not in self.global_reference.keys():
            warnings.append("Warning: No substrate reference found in global references.")
        for series, temps in self.filename_groups.items():
            for temp, data in temps.items():
                if 'sample' not in data.keys():
                    warnings.append(f"Warning: No sample found for series {series} at temperature {temp}.")
                if 'reference' not in data.keys():
                    warnings.append(f"Warning: No reference found for series {series} at temperature {temp}.")
        
        if warnings:
            for warning in warnings:
                print(warning)
        else:
            print("Integrity check passed: All groups have required components.")
        

    def _separate_by_delimiters(self, delimiter=None):
        '''Separates a filename into components based on provided delimiters.'''
        
        if not delimiter:
            delimiter = self.delimiter

        keyword_components = {}

        for filename in self.filelist:
            components = filename.split(delimiter)
            components = [comp.strip() for comp in components if comp.strip()] # remove empty strings and whitespace
            return components

    def simple_grouping(self, filename_grouping=None, substrate_key='substrate', ordering=None):
        '''Groups filenames based on a provided grouping dictionary. Intended as a quick soltion for a single format.
        
        filename_grouping: list of strings defining the order of components in the filename. E.g. ['type', 'series', 'temp']
        '''

        global_reference = {}

        if not filename_grouping:
            filename_grouping = ['type', 'series', 'temp']

        if not ordering:
            ordering = ['series', 'temp', 'type']

        
        series_index = filename_grouping.index('series')
        type_index = filename_grouping.index('type')
        temp_index = filename_grouping.index('temp')

        for filename in self.filelist:
            if 'reference' and 'air' in filename.lower(): # special case for air reference
                global_reference['air'] = filename
                continue
            # Isolate components
            details = '.'.join(filename.split('.')[:-1]) # Remove file extension

            details = details.split(self.delimiter)
            series = details[series_index]
            ftype = details[type_index]
            temp = details[temp_index].lower().strip(' k')

            if ftype == 'reference' and series == 'substrate':
                if 'substrate' not in global_reference:
                    global_reference['substrate'] = {temp: filename}
                else:
                    global_reference['substrate'][temp] = filename
                continue

            if len(details) > len(filename_grouping): # append any remaining details back to series
                extra = details[len(filename_grouping):]
                print("Found additional details for {}. Appending remaining details to series".format(filename))
                series = f"{series}_{self.delimiter.join(extra)}"

            if series not in self.filename_groups:
                self.filename_groups[series] = {}
            if ftype not in self.filename_groups[series]:
                self.filename_groups[series][temp] = {}
            self.filename_groups[series][temp][ftype] = filename


        # Attach global references
        for series, temps in self.filename_groups.items():
            for temp, data in temps.items():
                if 'reference' not in data.keys():
                    if 'substrate' in global_reference and temp in global_reference['substrate']:
                        data['reference'] = global_reference['substrate'][temp]

        print("Completed simple grouping of filenames:")
        print(self.filename_groups)
        print(global_reference)
        breakpoint()  # For debugging purposes




    def _identify_groups(self, filenames):
        '''Identifies groups of filenames based on keywords and delimiters.'''
        
        