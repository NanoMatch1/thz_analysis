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
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\reflection_testing"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-08_CNT-paper"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\reflection_testing\all_comp\CNT"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\reflection_testing"
    # fileDir = r"C:\Users\Samuel\Data\THz\CNTs\CNT-5"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\reflection_testing\all_comp\Si"
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
    import sys

    def acquisition_editor(fileDir):
        import acquisition_editor
        acquisition_editor.process_directory(fileDir)
        print("Acquisition editing complete.")
        breakpoint()
    
    def plot_sigma(dataset):
        fig, ax = plt.subplots()
        show_snr_mask = True
        for filename, data_obj in thz._sample_items(dataset):
            freq = data_obj.processing_dict.get('fft_freq')
            # n = data_obj.processing_dict.get('n')
            # k = data_obj.processing_dict.get('k')
            sigma = data_obj.processing_dict.get('sigma')
            real = sigma.real
            imag = sigma.imag
            if freq is None or real is None or imag is None:
                continue
            mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
            thz._plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, real, mask, label="{} (real)".format(filename))
            thz._plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, imag, mask, label="{} (imag)".format(filename))
        
        plt.legend()
        plt.xlabel("Frequency (THz)")
        # plt.ylabel("Refractive Index / Extinction Coefficient")
        plt.ylabel("Conductivity (S/m)")
        plt.title("Derived Conductivity")
        plt.show()
        return 
    

    ### --- Config Setup ---

    ### --- Acquisition editing ---

   
    def preprocess(dataset):
        dataset.load_all_data(case_insensitive=True)
        # dataset.plot_current()
        # --- For reflection data ---
        dataset.group_files(keywords=['type'])
        thz.subtract_baseline(dataset, show_graph=show_graph)
        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---

        # --- alignment step, different protocols
        thz.align_to_reference(dataset, ref_type="reference", roi=(152, 156))

        thz.normalise(dataset, config={"bounds": (152, 156)}, show_graph=show_graph)
        thz.segment_reflections(dataset, show_graph=True)

        # dataset.save_state()


    # acquisition_editor(fileDir)
    dataset = DataSet(fileDir)
    show_graph = True
    dataset.load_all_data(case_insensitive=True)
    dataset.plot_current()
        
    ### --- Pre-processing and segmentation ---
    # preprocess(dataset)
    thz.subtract_baseline(dataset, show_graph=show_graph)
    dataset.group_files(keywords=['type', 'seri'])
    dataset.grouping.show_matches()

    thz.align_to_reference(dataset, ref_type="reference", subsample_correction=True, show_graph=show_graph)
    # dataset.plot_current()
    # dataset.load_database('CNT_interp_norm_2_db.pkl')
    # thz.segment_reflections(dataset, show_graph=show_graph)

    # dataset.group_files(keywords=['type'])

    # thz.pre_window_align_peak(dataset, show_graph=show_graph)
    # thz.plot_current(dataset)
    # thz.pre_window_align_peak(dataset,show_graph=show_graph)
    # thz.global_truncate(dataset)
    # dataset.plot_current()
    # breakpoint()

    # dataset.save_state()
    # show_graph = False
    # dataset.load_database()  # optional .db file with pre-parsed metadata; skip if you want to re-parse from the raw files
    # thz.result_viewer(dataset)
    # dataset.save_database()
    # print("Stop after pre-processing and alignment.")
    # breakpoint()
    thz.window_time(dataset, config={"window": {"type": "hann", "length": 0.3}}, show_graph=show_graph)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 3.0}}, show_graph=show_graph)
    thz.fft_spectrum(dataset)
    thz.trusted_band_mask(dataset, config={"mask": {"snr_thresh_db": 1.5,
                                                 "tail_fraction": 0.25,
                                                 "min_contiguous_bins": 3}})
    thz.transfer_function(dataset, config={"transfer": {"apply_snr_mask": True}}, ref_type='reference')
    # dataset.save_state()
    # dataset.save_database()

    # thz.extend_grid
    # thz.plot_fft(dataset, freq_range=(0.3, 10), normalise=True, scale='log')
    # thz.plot_fft(dataset)

    # thz.phase_correction(dataset, source='transfer')

    # dataset.save_state()
    # thz.invert_nk(dataset, thickness_m=1e-3)
    thz.invert_nk_reflection(
        dataset,
        geometry="window",
        theta_deg=45,
        polarization='s',
        n_window=1.95,
    )
    # thz.invert_nk_reflection(
    #     dataset,
    #     geometry="gold",
    #     theta_deg=45,
    #     polarization='s',
    #     # n_window=1.95,
    # )
    thz.derive_eps_sigma(dataset)

    thz.result_viewer(dataset)
    thz.export_results(dataset)