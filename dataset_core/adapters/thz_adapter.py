import thz_core as core
from dataset_core.dataset import DataSet, DataService
"""Used to bridge the DataSet manager and the thz analysis library."""

def align_on_peak(dataset: DataSet, show_graph: bool = False, auto_range: tuple = None) -> DataSet:
    '''Aligns all acquisitions in the dataset on their main peak.'''

    data_dict = dataset.data.data_dict
    new_dict = {}
    for filename, data_obj in data_dict.items():
        new_dict[filename] = data_obj.data.copy()

    new_dict = core.align_on_peak(new_dict, auto_range=auto_range)

    if show_graph:
        import matplotlib.pyplot as plt
        for filename, data in new_dict.items():
            plt.plot(data[:, 1], label=filename)
        plt.legend()
        plt.show()

    # modifies the dataset in place, returns for convenience
    for filename, data_obj in dataset.data.data_dict.items():
        breakpoint()
        data_obj.data = new_dict[filename]

    return dataset