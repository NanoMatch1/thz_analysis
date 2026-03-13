from dataset_core.dataset import DataSet


if __name__ == "__main__":
    fileDir = r"C:\Users\Samuel\matchbook\thz\dataset_core\example_data"

    dataset = DataSet(fileDir)
    dataset.load_all_data()
    # dataset.plot_current()
    edited = dataset.modify_acquisitions(in_place=True, export=True)
    dataset.plot_current()
    