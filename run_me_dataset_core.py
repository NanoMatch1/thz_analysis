from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


if __name__ == "__main__":
    fileDir = r"C:\Users\Samuel\matchbook\thz\dataset_core\example_data"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-25_MINTS"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2"

    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)

    fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2\export"
    dataset = DataSet(fileDir)
    # dataset.load_all_data()
    # # dataset.plot_current()

    # # edited = dataset.modify_acquisitions(in_place=True, export=True)
    # dataset.group_files(keywords=['type', 'series'])
    # dataset.grouping.show_pairs()

    dataset.load_state()
    thz.align_on_peak(dataset, show_graph=True, auto_range=(50,60))
    dataset.save_state()

    breakpoint()

    dataset.plot_current()