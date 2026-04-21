import os

import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools


if __name__ == "__main__":
    fileDir = r"C:\Users\Samuel\matchbook\thz\dataset_core\example_data"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-23_MINTS"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-01-22_P6-AERO\export"
    # fileDir = r"C:\Users\Samuel\Data\THz\diagnostics\2026-04-09_new-STE"
    # fileDir = r"C:\Users\Samuel\Data\THz\Co_HHTP_Tdep_TDS"
    fileDir = r"C:\Users\Samuel\Data\THz\Co_HHTP_Tdep_TDS\analysis"
    fileDir = r"C:\Users\Samuel\Data\THz\Ni_HHTP_Tdep_TDS"
    fileDir = r"C:\Users\Samuel\Data\THz\M-HHTP_Crossover_Tdep\2026-04-21_stage_adjustment_test\test airs"

    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)

    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2\export"
    # fileDir = r"C:\Users\Samuel\Data\Chris"
    dataset = DataSet(fileDir)

    def preprocess(dataset):
        dataset.load_all_data(case_insensitive=True)
        # dataset.data_dict

        for filename, data_obj in dataset.data_dict.items():
            print(f"{filename}")
            # breakpoint()
        # thz.fft_spectrum(dataset)
        # dataset.plot_current()
        # thz.plot_fft(dataset)
        # edited = dataset.modify_acquisitions(in_place=True, export=True)
        # validation = thz.validate_thz(dataset, verbose=True, label="Input Validation", permit=["clipping"])
        # breakpoint()
        # thz.print_metrics(validation)
        dataset.group_files(keywords=['type', 'temperature'])
        dataset.grouping.show_pairs()
        breakpoint()
        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---
        thz.subtract_baseline(dataset)
        # thz.align_on_peak(dataset, show_graph=True)#, auto_range=(40,60))
        dataset.save_state()

    preprocess(dataset)
    dataset.load_state()
    thz.window_time(dataset, config={"window": {"type": "hann", "alpha": 0.25}}, show_graph=False)

    # thz.plot_current(dataset)
    
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
    # thz.extend_grid
    thz.fft_spectrum(dataset)

    thz.plot_fft(dataset)
    # thz.plot_fft(dataset)

    # breakpoint()
    


    thz.transfer_function(dataset)
    from thz_core import invert_nk
    from matplotlib import pyplot as plt
    import numpy as np
    # for filename, data in dataset.data.items():
    #     phase_data = data.data
    #     # for index in range(*phase_range):
    #     plt.plot(data.data[:, 0], np.unwrap(data.data[:, 2]), label="Original Phase {}".format(filename))
    # plt.legend()
    # plt.show()
    # # phase_after = tools.inspect_phase(dataset, title="Phase After Transfer Function")
    # # new_dataset = 

    # phase_dict = {}
    # for filename, data in dataset.data.items():
    #     phase_dict[filename] = data.data
    #     # plt.plot(data.data[:, 0], data.data[:, 1], label="Amplitude {}".format(filename))


    # offset_dict = {}
    # for filename, phase_data in phase_dict.items():
    #     if 'reference' in filename.lower():
    #         continue
    #     for offset in range(5):
    #         offset_phase = tools.phase_offset(phase_data, offset=offset)
    #         # plt.plot(phase_data[:, 0], offset_phase, label="Offset Phase {} Offset {}".format(filename, offset))
    #         if filename not in offset_dict:
    #             offset_dict[filename] = {}
    #         offset_dict[filename][offset] = np.column_stack((phase_data[:, 0], phase_data[:, 1], offset_phase))

    # fig, ax = plt.subplots(3, 1)
    # for filename, offsets in offset_dict.items():
    #     for value, data in offsets.items():
    #         mask = np.array([True if 0.3e12 < x < 3e12 else False for x in data[:, 0]])
    #         data = data[mask]
    #         n, k, metrics = invert_nk(data[:, 0], data[:, 1], thickness_m=1e-3, mask=[True for _ in range(data.shape[0])], config=None)
    #         ax[0].plot(data[:, 0], n, label="n {} Offset {}".format(filename, value))
    #         ax[1].plot(data[:, 0], k, label="k {} Offset {}".format(filename, value))
    #         ax[2].plot(data[:, 0], data[:, 2], label="Phase {} Offset {}".format(filename, value))
    #     ax[0].legend()
    #     ax[1].legend()
    #     ax[2].legend()
    #     plt.show()
    # breakpoint()
    
    # for filename, offsets in offset_dict.items():
    # breakpoint()


    # thz.phase_correction_demo(dataset, source='transfer')

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