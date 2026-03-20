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

    def preprocess(dataset):
        dataset.load_all_data()
        # edited = dataset.modify_acquisitions(in_place=True, export=True)
        dataset.group_files(keywords=['type', 'series'])
        dataset.grouping.show_pairs()

        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---
        thz.subtract_baseline(dataset)
        thz.align_on_peak(dataset, show_graph=True, auto_range=None)
        dataset.save_state()

    preprocess(dataset)
    # dataset.load_state()
    thz.window_time(dataset, config={"type": "tukey", "alpha": 0.25})
    # thz.zero_pad(dataset, config={"extend_factor": 2.0})
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset)
    # dataset.save_state()
    thz.invert_nk(dataset, thickness_m=1e-3)
    thz.derive_eps_sigma(dataset)


    # breakpoint()



    # dataset.plot_current()
    thz.result_viewer(dataset)