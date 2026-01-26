Data Structures:

DataSet orchestrates the overall processing of the dataset, and handles batch processing of whatever files you give it. It contains a .data attribute which is a DataService service class. 

The DataService holds all the data within a dictionary, and can be accessed by iterating over the object, or through the class attribute ._data_dict.

A GroupingService performs grouping related actions, such as pairing of references with samples. This is instantiated by the DataService, and accessed through data.grouping. Calling the grouping service with a filename argument returns the grouping information for that filename. Note a grouping function must first be called on the dataset through the grouping service to identify reference and filename keywords from the filename.
Grouping can be adjusted dynamically and modified to group files based on a series, such as sample, or temperature. This allows quick access of database or pivot table-like characteristics.

Specific data types are held in DataStructures, and coordinate the use of analysis functions, such as fft and baseline methods. Their role is to organise the data, while analysis is handled via the analytical methods to preserve separation of tasks.

