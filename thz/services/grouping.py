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


    def __init__(self, filename: str, keywords=['type', 'series', 'temp'],**kwargs):
        self.report_list = ['data_type', 'series']
        self.filename = filename
        self.series = None
        self.data_type = None # 'sample' or 'reference'
        self.keywords = keywords

        self.extra_details = None
        self.air_reference = None
        self.substrate_reference = None

        self._parse_filename(keywords=self.keywords, **kwargs)

    def __repr__(self):
        # details =
        infolist = []
        for key in self.report_list:
            value = self.__dict__.get(key, None)
            infolist.append(f" -> {key}: {value}")
        
        for key in ['extra_details', 'air_reference', 'substrate_reference']:
            infolist.append(f" -> {key}: {self.__dict__.get(key, None)}")
        
        return f"Filename: {self.filename}\n" + "\n".join(infolist)
    
    def __str__(self):
        return self.filename

    def _parse_filename(self, delimiter: str = '_', keywords: list = ['type', 'series', 'temp'], **kwargs):
        '''Parses the filename into components based on provided delimiter and keywords order.'''

        details = '.'.join(self.filename.split('.')[:-1]) # Remove file extension
        components = details.split(delimiter)
        components = [comp.strip() for comp in components if comp.strip()] # remove empty strings and whitespace

        for index, key in enumerate(keywords):
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
                    self.report_list.append(key) # add new keyword report list

        # handle any extra components *after* assigning main fields
        if len(components) > len(keywords):
            extras = components[len(keywords):]
            if kwargs.get('merge_extra', False):
                self.series = delimiter.join([self.series, *extras])
            self.extra_details = extras
            

class GroupingService:

    '''Creates a nested dictionary structure to group filenames based on provided delimiters and keywords.
    
    Handles tracking of the current dataset through modifications to the _current_data_list attribute. DataSet can access and modify this attribute to control which files are being worked on.'''

    def __init__(self, keywords=['type', 'series', 'temp'], delimiter='_', **kwargs):
        self.filelist = kwargs.get('filelist', [])
        self.file_items = {}
        self.keywords = keywords
        self.filename_groups = []
        self.global_reference = {}
        self.keywords = keywords
        self.delimiter = delimiter
        self.__dict__.update(kwargs)

        self._current_data_list = []

    def is_reference(self, filename):
        item = self.file_items.get(filename, None)
        if item is None:
            return False
        return item.data_type == 'reference'

    def is_sample(self, filename):
        item = self.file_items.get(filename, None)
        if item is None:
            return False
        return item.data_type == 'sample'

    @property
    def info(self):
        print(f"GroupingService with {len(self.filelist)} files.")
        print(f"Current grouping keywords: {self.keywords}")
        print(f"Current delimiter: '{self.delimiter}'")
        print(f"Number of filename groups: {len(self.filename_groups)}")

    @property
    def elaborate(self):
        for filename, item in self.file_items.items():
            print(item.__repr__())

    def set_grouping_keywords(self, new_keywords):
        self.keywords = new_keywords

    def get_current_data_list(self):
        return self._current_data_list
    
    def set_current_data_list(self, new_list):
        self._current_data_list = new_list

    def __call__(self, filename, object_type=None):
        file_obj = self.file_items.get(filename, None)
        if file_obj is None:
            print(f"Filename {filename} not found in file items.")
            return None
        
        if object_type is None:
            return file_obj
        
        if hasattr(file_obj, object_type):
            return getattr(file_obj, object_type)

        return
    
    def get_reference_filename(self, filename, ref_type='substrate'):
        '''Finds the corresponding reference filename for a given sample filename based on grouping.'''
        if ref_type not in ['substrate', 'air']:
            raise ValueError("ref_type must be either 'substrate' or 'air'.")
        
        group_info = self(filename)
        if group_info is None:
            print(f"Grouping info not found for filename: {filename}")
            return None

        data_type = group_info.data_type

        if data_type == 'reference':
            return None  # No reference for a reference file
        elif data_type == 'sample':
            if ref_type == 'air':
                reference_filename = group_info.air_reference
            else:
                reference_filename = group_info.substrate_reference
            return reference_filename
        else:
            print(f"Unknown data type '{data_type}' for filename: {filename}")
            return None

    
    def update(self, filelist=[]):
        '''Updates the grouping service with new filelist by appending to the old, and rebuilds file items.'''

        for filename in filelist:
            self.filelist.append(filename)
        
        self._current_data_list = self.filelist # note this is a shallow copy, i.e. a reference to filelist, and will follow any changes made to it.
        # self._build_fileitems()

    def _build_fileitems(self, **kwargs):
        '''Builds FilenameItem objects for each filename in the filelist.'''

        for filename in self.filelist:
            item = FilenameItem(filename, **kwargs)
            self.file_items[filename] = item

    def parse_filenames(self, **kwargs):
        '''Parses all filenames in file_items using provided delimiters and grouping order.'''

        # Use existing keywords if none provided
        if kwargs.get('keywords', None) is None:
            keywords = self.keywords
            kwargs['keywords'] = keywords

        for item in self.file_items.values():
            item._parse_filename(**kwargs)

    def _group_by_keyword(self, keyword):
        '''Tries to return a dictionary of files grouped by the keyword, if the keyword is present in the filename and itentified by the grouping service.'''

        grouping_dict = {}

        for filename, fileitem in self.file_items.values():
            key_value = getattr(fileitem, keyword, None)
            if key_value is None:
                continue
            if key_value not in grouping_dict:
                grouping_dict[key_value] = {filename: fileitem}
            else:
                grouping_dict[key_value][filename] = fileitem
            
        return grouping_dict
    
    def _identify_global_references(self):
        for filename, item in self.file_items.items():
            if item.data_type == 'reference' and 'air' in item.series.lower(): # special case for air reference
                self.global_reference['air'] = item.filename 
                continue
            elif item.data_type == 'reference' and 'substrate' in item.series.lower(): # special case for substrate reference
                self.global_reference['substrate'] = item.filename 
                continue
  

    def simple_grouping(self, delimiter='_', keywords=['type', 'series', 'temp']):
        '''Groups data by slicing the filename. Expects filename to contain data outlined in keywords, and does not (yet) logically check those parameters.
        
        currently: keywords: list of strings defining the order of components in the filename. E.g. ['type', 'series', 'temp']

        # TODO: change logic to handle type independently, then grouping is performed on these separately, and on the rest of the keywords.
        # TODO: Pair references based on filename inclusive of delimiters after type, not just temperature.
        '''

        self._build_fileitems(delimiter=delimiter, keywords=keywords, merge_extra=True)
        self.parse_filenames()
        self._identify_global_references()
        print("Completed simple grouping of filenames.")

        self.integrity_check()
        self._pair_references()

    def integrity_check(self):
        '''Performs an integrity check on the grouped filenames to ensure each group has the expected components required for analysis.'''

        warnings = []

        if 'air' not in self.global_reference.keys():
            warnings.append("Warning: No air reference found in global references.")
        if 'substrate' not in self.global_reference.keys():
            warnings.append("Warning: No substrate reference found in global references.")
        # for series, temps in self.filename_groups.items():
        #     for temp, data in temps.items():
        #         if 'sample' not in data.keys():
        #             warnings.append(f"Warning: No sample found for series {series} at temperature {temp}.")
        #         if 'reference' not in data.keys():
        #             warnings.append(f"Warning: No reference found for series {series} at temperature {temp}.")
        
        if warnings:
            for warning in warnings:
                print(warning)
        else:
            print("Integrity check passed: All groups have required components.")
        
    def _pair_references(self):
        '''Works through file_items to pair reference and samples based on the number of unique grouping keyword identifiers. Currently matches on all provided keywords except 'data_type' and 'series'. If no extra keywords are provided, matches all samples to global substrate reference.'''

        references = {filename: item for filename, item in self.file_items.items() if item.data_type == 'reference'}
        samples = {filename: item for filename, item in self.file_items.items() if item.data_type == 'sample'}
        
        for filename, fileitem in samples.items():
            keywords = fileitem.report_list.copy()
            keywords.remove('data_type')  # remove type to match on other keywords
            keywords.remove('series')  # remove series to match on other keywords 
            match_criteria = {key: getattr(fileitem, key, None) for key in keywords}

            if match_criteria == {}:
                # if no extra keywords to match on, assign global substrate reference
                fileitem.substrate_reference = self.global_reference.get('substrate', None)
                continue

            # Find matching reference
            for ref_filename, ref_item in references.items():
                match = True
                for key, value in match_criteria.items():
                    if getattr(ref_item, key, None) != value:
                        match = False
                        break
                if match:
                    if ref_item.data_type == 'reference':
                        fileitem.substrate_reference = ref_filename
                        fileitem.air_reference = self.global_reference.get('air', None)

    def _separate_by_delimiters(self, delimiter=None):
        '''Separates a filename into components based on provided delimiters.'''
        
        if not delimiter:
            delimiter = self.delimiter

        keyword_components = {}

        for filename in self.filelist:
            components = filename.split(delimiter)
            components = [comp.strip() for comp in components if comp.strip()] # remove empty strings and whitespace
            return components

    @property
    def references(self):
        '''Returns a dictionary of all identified references in the file items.'''
        references = {}
        for filename, item in self.file_items.items():
            if item.data_type == 'reference':
                references[filename] = item
        return references

    @property
    def samples(self):
        '''Returns a dictionary of all identified samples in the file items.'''
        samples = {}
        for filename, item in self.file_items.items():
            if item.data_type == 'sample':
                samples[filename] = item
        return samples

        