'''Service for grouping of filenames based on strings, keywords and delimiters. Used to correlate reference and sample across variable datasets suhch as temperature series.'''


class GroupService:

    '''Creates a nested dictionary structure to group filenames based on provided delimiters and keywords.'''

    def __init__(self, filelist, keywords=[''], delimiter='_', **kwargs):
        self.filelist = filelist
        self.filename_groups = {}
        self.keywords = keywords
        self.delimiter = delimiter
        self.__dict__.update(kwargs)

    def _separate_by_delimiters(self, delimiter=None):
        '''Separates a filename into components based on provided delimiters.'''
        
        if not delimiter:
            delimiter = self.delimiter

        keyword_components = {}

        for filename in self.filelist:
            components = filename.split(delimiter)
            components = [comp.strip() for comp in components if comp.strip()] # remove empty strings and whitespace
            return components
            


    def _identify_groups(self, filenames):
        '''Identifies groups of filenames based on keywords and delimiters.'''
        
        