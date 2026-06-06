import os

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools


def assess_drift(dataset):
    '''Compiles the drift at max intensity across all scans and plots it to visualize any trends.'''
    for filename, data_obj in dataset.data_dict.items():
        drift = data_obj.assess_drift()
        plt.plot(drift, label=filename)
    plt.legend()
    plt.title("Drift Assessment")
    plt.xlabel("Acquisitions")
    plt.ylabel("Amplitude at Max Intensity Point")
    plt.show()



def decompose_acc_file(filepath):
    '''Loads a .acc file and decomposes it into its constituent parts: metadata, time grid, and acquisition data. Returns a dictionary with these components.'''
    for filename, data_obj in dataset.data_dict.items():
        print(f"{filename}")
        dataX = data_obj.raw_data[:, 0]
        breakpoint()
        for index in range(data_obj.raw_data.shape[1]):
            if index == 0:
                continue                
            new_filename = "air_LN2_filling_t-{}".format(index)
            dataY = data_obj.raw_data[:, index]
            new_dict[new_filename] = np.column_stack((dataX, dataY))

    for filename, data in new_dict.items():
        np.savetxt(os.path.join(fileDir, filename + ".acc"), data)


if __name__ == "__main__":
    fileDir = r"C:\Users\Samuel\matchbook\thz\dataset_core\example_data"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-23_MINTS"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-01-22_P6-AERO\export"
    # fileDir = r"C:\Users\Samuel\Data\THz\diagnostics\2026-04-09_new-STE"
    # fileDir = r"C:\Users\Samuel\Data\THz\Co_HHTP_Tdep_TDS"
    # fileDir = r"C:\Users\Samuel\Data\THz\Co_HHTP_Tdep_TDS\analysis"
    # fileDir = r"C:\Users\Samuel\Data\THz\Ni_HHTP_Tdep_TDS"
    # fileDir = r"C:\Users\Samuel\Data\THz\M-HHTP_Crossover_Tdep\2026-04-21_stage_adjustment_test\test airs"
    # fileDir = r"C:\Users\Samuel\Data\THz\M-HHTP_Crossover_Tdep\filling_test"
    # fileDir = r"C:\Users\Samuel\Data\THz\M-HHTP_Crossover_Tdep\Cu_HHTP_Tdep_TDS"
    fileDir = r"C:\Users\Samuel\Data\THz\diagnostics\2026-05-04_ZnTe-STE"
    fileDir = r"C:\Users\Samuel\Data\THz\diagnostics\2026-05-05_ste-test\test"
    fileDir = r"C:\Users\Samuel\Data\THz\Reference Data"
    fileDir = r"C:\Users\Samuel\Data\THz\CNTs\Sam\Sam\data\test"
    fileDir = r"C:\Users\Samuel\Data\THz\M-HHTP_Crossover_Tdep\2026-06-02_TDS-ambients\Comparison\export\export"
    fileDir = r"C:\Users\Samuel\Data\THz\CNTs\2026-06-03_CNT-4\analysis"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-paper\export"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-paper\testing"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-paper\testing\segmented\second_reflection"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-bare"
    # fileDir = r"C:\Users\Samuel\Data\THz\M-HHTP_Crossover_Tdep\2026-04-21_Co_HHTP_Tdep_TDS"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-04_OPTP-M-HHTP"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-paper\export"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_OPTP-M-HHTP"
    # fileDir = r"C:\Users\Samuel\Data\THz\diagnostics\2026-05-08_noise"

    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)
    # acquisition_editor.convert_directory(fileDir)

    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2\export"
    # fileDir = r"C:\Users\Samuel\Data\Chris"
    dataset = DataSet(fileDir)
    show_graph = True

    # use_reference = 'gold'

    # dataset.load_state()

    # dataset.plot_current()
    
    def preprocess(dataset):
        dataset.load_all_data(case_insensitive=True)
        dataset.plot_current()
        # --- For reflection data ---
        # thz.segment_reflections(dataset, show_graph=True)
        dataset.group_files(keywords=['type'])
        thz.align_to_reference(dataset, show_graph=True)
        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---
        # thz.subtract_baseline(dataset, show_graph=show_graph)
        # thz.align_on_peak(dataset, show_graph=True)#, auto_range=(40,60))
        dataset.save_state()

    preprocess(dataset)
    # breakpoint()
    # dataset.group_files(keywords=['type', 'temp', 'set', 'extra'])
    dataset.load_state()
    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()
    # segment_dict = {"first_segment": (151, 158.10), "second_segment": (157.9, 162.8)}
    thz.plot_current(dataset)

    thz.window_time(dataset, config={"window": {"type": "hann", "length": 0.2}}, show_graph=show_graph)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}}, show_graph=show_graph)
    # thz.extend_grid
    thz.fft_spectrum(dataset)
    # thz.plot_fft(dataset, freq_range=(0.3, 10), normalise=True, scale='')
    # thz.plot_fft(dataset)
    thz.transfer_function(dataset, ref_type='reference')

    # thz.phase_correction(dataset, source='transfer')

    # dataset.save_state()
    # thz.trusted_band_mask(dataset, config={"mask": {"snr_thresh_db": 2,
    #                                              "tail_fraction": 0.25,
    #                                              "min_contiguous_bins": 3}})
    # thz.invert_nk(dataset, thickness_m=1e-3)
    thz.invert_nk_reflection(
        dataset,
        geometry="gold",
        theta_deg=45,
        polarization='s',
        # n_window=N_SIO2,
    )
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