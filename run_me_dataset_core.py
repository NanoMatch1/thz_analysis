from dataset_core.dataset import DataSet
import thz_core as thz


if __name__ == "__main__":
    fileDir = r"C:\Users\Samuel\matchbook\thz\dataset_core\example_data"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-25_MINTS"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2"

    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)

    fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2\export"
    dataset = DataSet(fileDir)
    dataset.load_all_data()
    # dataset.plot_current()
    print(thz.__all__)
    # edited = dataset.modify_acquisitions(in_place=True, export=True)
    dataset.group_files(keywords=['type', 'series'])
    dataset.grouping.show_pairs()
    # breakpoint()
    # dataset.align_acquisitions(in_place=True, export=True)
    dataset.plot_current()