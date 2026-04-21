'''Service for grouping of filenames based on strings, keywords and delimiters. Used to correlate reference and sample across variable datasets suhch as temperature series.'''

#TODO - create dataset module import, others can import from dataset_core.dataset 

from dataclasses import dataclass, field
from typing import Optional
from dataset_core.data_structures.filename_info import FilenameInfo

@dataclass
class TemperatureItem:
    """Placeholder for identifying a temperature-dependent measurement from the filename.

    Constructed from filename parsing. Extend with additional fields
    (e.g. unit, measurement_label) as needed for your analysis."""

    temperature_value: Optional[float] = None
    temperature_unit: str = 'K'
    filename: str = ''


# Backwards-compatible alias. Prefer FilenameInfo in new code.
FilenameItem = FilenameInfo
            

class GroupingService:

    '''Creates a nested dictionary structure to group filenames based on provided delimiters and keywords.
    
    Handles tracking of the current dataset through modifications to the _current_data_list attribute. DataSet can access and modify this attribute to control which files are being worked on.'''

    reference_identifiers = ['reference', 'ref', 'substrate', 'air']
    sample_identifiers = ['sample']

    def __init__(self, keywords=None, delimiter='_', filelist=None):
        self.filelist = filelist if filelist is not None else []
        self.file_items = {}
        self.keywords = keywords if keywords is not None else ['type', 'series', 'temp']
        self.filename_groups = []
        self.global_reference = {}
        self.delimiter = delimiter

        self._current_data_list = []

    def help(self):
        help_text = """
        grouping_service = GroupingService(keywords=['type', 'series', 'temp'], delimiter='_')
        grouping_service.update(filelist)
        grouping_service.simple_grouping()
        reference_file = grouping_service.get_reference_filename(sample_filename, ref_type='substrate')
        grouping_service.is_reference(filename)
        grouping_service.is_sample(filename)
        """
        print(help_text)

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
    def info(self) -> str:
        '''Returns a summary string describing the GroupingService state.'''
        lines = [
            f"GroupingService with {len(self.filelist)} files.\n",
            f"Current grouping keywords: {self.keywords}\n",
            f"Current delimiter: '{self.delimiter}'\n",
            f"Number of filename groups: {len(self.filename_groups)}\n",
        ]
        return lines

    @property
    def elaborate(self) -> str:
        '''Returns a detailed string representation of all file items.'''
        lines = [item.__repr__() for item in self.file_items.values()]
        return lines
    
    def show_pairs(self):
        """Shows the current sample-reference pairs based on the grouping. Prints the filename of each sample along with its assigned substrate and air reference filenames."""
        
        for filename, item in self.file_items.items():
            if item.data_type == 'sample':
                substrate_ref = item.substrate_reference if hasattr(item, 'substrate_reference') else 'None'
                air_ref = item.air_reference if hasattr(item, 'air_reference') else 'None'
                print(f"Sample: {filename}")
                print(f"  > Substrate Ref: {substrate_ref}")
                print(f"  > Air Ref: {air_ref}")

    def get_state(self):
        '''Returns a dictionary representing the current state of the grouping service, including file items and grouping keywords.'''
        state = {
            "file_items": self.file_items,
            "keywords": self.keywords,
            "delimiter": self.delimiter,
            "global_reference": self.global_reference,
            "filelist": list(self.filelist),
            "current_data_list": list(self._current_data_list),
            "get_state": self.get_state,  # include method for reference
        }
        return state
    
    def restore_state(self, state):
        '''Restores the grouping service state from a provided dictionary.'''
        self.file_items = state.get("file_items", {})
        self.keywords = state.get("keywords", self.keywords)
        self.delimiter = state.get("delimiter", self.delimiter)
        self.global_reference = state.get("global_reference", self.global_reference)
        self.filelist = state.get("filelist", list(self.file_items.keys()))
        self._current_data_list = state.get("current_data_list", self.filelist)

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

    
    def update(self, filelist=None):
        '''Updates the grouping service with new filelist by appending to the old, and rebuilds file items.'''

        if filelist is None:
            filelist = []

        for filename in filelist:
            self.filelist.append(filename)
        
        self._current_data_list = self.filelist # note this is a shallow copy, i.e. a reference to filelist, and will follow any changes made to it.
        # self._build_fileitems()

    def _build_fileitems(self, **kwargs):
        '''Builds FilenameItem objects for each filename in the filelist.'''

        for filename in self.filelist:
            item = FilenameInfo.from_filename(filename, **kwargs)
            self.file_items[filename] = item

    def parse_filenames(self, **kwargs):
        '''Parses all filenames in file_items using provided delimiters and grouping order.'''

        # Use existing keywords if none provided
        if kwargs.get('keywords', None) is None:
            keywords = self.keywords
            kwargs['keywords'] = keywords

        for item in self.file_items.values():
            item.parse(**kwargs)

    def _group_by_keyword(self, keyword):
        '''Tries to return a dictionary of files grouped by the keyword, if the keyword is present in the filename and itentified by the grouping service.'''

        grouping_dict = {}

        for filename, fileitem in self.file_items.items():
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
  

    def simple_grouping(self, delimiter: str = '_', keywords: list | None = None):
        '''Groups data by slicing the filename. Expects filename to contain data outlined in keywords, and does not (yet) logically check those parameters.
        
        currently: keywords: list of strings defining the order of components in the filename. E.g. ['type', 'series', 'temp']

        # TODO: change logic to handle type independently, then grouping is performed on these separately, and on the rest of the keywords.
        # TODO: Pair references based on filename inclusive of delimiters after type, not just temperature.
        '''

        selected_keywords = keywords if keywords is not None else self.keywords

        self._build_fileitems(delimiter=delimiter, keywords=selected_keywords, merge_extra=True)
        self.keywords = selected_keywords
        self.parse_filenames()
        # self._identify_global_references()
        print("Completed simple grouping of filenames.")

        self.integrity_check()
        self._pair_references()
        
        return self.file_items

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
        '''Works through file_items to pair reference and samples based on the number of unique grouping keyword identifiers. Currently matches on all provided keywords except 'data_type' and 'series'. If no extra keywords are provided, matches all samples to global substrate reference.
        If matches are found to a reference, adds attributes to the sample with that reference name - e.g. substrate_reference, air_reference... Constructed from the keyword, so any keyword can become a reference type internally.'''

        references = {filename: item for filename, item in self.file_items.items() if item.data_type in self.reference_identifiers}
        samples = {filename: item for filename, item in self.file_items.items() if item.data_type in self.sample_identifiers}

        # for each sample, establish match criteria and find references
        for filename, fileitem in samples.items():
            keywords = fileitem.report_list.copy()
            keywords.remove('data_type')  # remove type to match on other keywords
            # keywords.remove('series')  # remove series to match on other keywords 
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
                    reftype = ref_item.data_type
                    fileitem.__dict__[f"{reftype}_reference"] = ref_filename
                    breakpoint()

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
