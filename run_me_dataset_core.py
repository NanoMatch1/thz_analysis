import os

import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools


if __name__ == "__main__":
    fileDir = r"C:\Users\Samuel\matchbook\thz\dataset_core\example_data"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-23_MINTS"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-01-22_P6-AERO\export"

    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)

    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2\export"
    # fileDir = r"C:\Users\Samuel\Data\Chris"
    dataset = DataSet(fileDir)

    def preprocess(dataset):
        dataset.load_all_data()
        # edited = dataset.modify_acquisitions(in_place=True, export=True)
        validation = thz.validate_thz(dataset, verbose=True, label="Input Validation")
        # thz.print_metrics(validation)
        dataset.group_files(keywords=['type', 'series'])
        dataset.grouping.show_pairs()

        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---
        thz.subtract_baseline(dataset)
        thz.align_on_peak(dataset, show_graph=True)#, auto_range=(40,60))
        dataset.save_state()

    # preprocess(dataset)
    dataset.load_state()
    thz.window_time(dataset, config={"window": {"type": "tukey", "alpha": 0.3}}, show_graph=False)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 4.0}})
    thz.fft_spectrum(dataset)

    phase_before = tools.inspect_phase(dataset, title="Phase Before Unwrapping")

    thz.transfer_function(dataset)
    phase_after = tools.inspect_phase(dataset, title="Phase After Transfer Function")

    # dataset.save_state()
    thz.trusted_band_mask(dataset, config={"mask": {"snr_thresh_db": 6,
                                                 "tail_fraction": 0.25,
                                                 "min_contiguous_bins": 3}})
    thz.invert_nk(dataset, thickness_m=1e-3)
    thz.derive_eps_sigma(dataset)


    # breakpoint()
    thz.export_results(dataset)
    # from matplotlib import pyplot as plt

    # for filename, data in dataset.data.items():
    #     # print(f"{filename}: {data.keys()}")
    #     if 'reference' in filename.lower():
    #         continue
    #     for key, results in data.processing_dict.items():
    #         print(key)
    #         print(results)
    #         breakpoint()

    # breakpoint()

    # dataset.plot_current()
    thz.result_viewer(dataset)