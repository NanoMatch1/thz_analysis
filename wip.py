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
    import sys

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
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-10"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-10\CNT-10B"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-10\CNT-10B\segmented\second_reflection"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-11\A\segmented\second_reflection"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-11\B\segmented\second_reflection"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-12\segmented\second_reflection"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\A"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-14\A"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-14\A\segmented\second_reflection"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\A\segmented\second_reflection"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\B"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\B\segmented\second_reflection"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-12\B\segmented\second_reflection"


    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)
    # acquisition_editor.convert_directory(fileDir)

    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\MINTS_batch-2\export"
    # fileDir = r"C:\Users\Samuel\Data\Chris"

    def filter_dataset(dataset, keyword, set_current=True):
        """Filter the dataset to include only files that contain the specified keyword in their filename."""
        data_dict = {filename: data_obj for filename, data_obj in dataset.data_dict.items() if keyword in filename}
        if set_current:
            dataset.set_current_files(data_dict.keys())
        return data_dict

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
   
    def preprocess(dataset, show_graph=False):
        dataset.load_all_data(case_insensitive=True)
        # dataset.plot_current()
        # --- For reflection data ---
        dataset.group_files(keywords=['type'])
        thz.subtract_baseline(dataset, show_graph=show_graph)
        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---

        # --- alignment step, different protocols
        thz.align_to_reference(dataset, ref_type="reference", roi=(152, 156), subsample_correction=True, show_graph=show_graph)
        # dataset.plot_current()

        thz.normalise(dataset, config={"bounds": (152, 156)}, show_graph=show_graph)
        thz.segment_reflections(dataset, show_graph=True, segments={'first_reflection': (151, 158.8), 'second_reflection': (163, 170)})
        print("Pre-processing complete.")
        sys.exit()

        # dataset.save_state()


    # acquisition_editor(fileDir)
    ### --- Pre-processing and segmentation ---
    dataset = DataSet(fileDir)
    show_graph = False
    dataset.load_all_data(case_insensitive=True)
    # dataset.plot_current()
    # --- Preprocess reflections - uncomment to segment and normalise time traces. Comment and re-run with segmented folder to continue analysis
    # preprocess(dataset)
    # ---

    thz.subtract_baseline(dataset, show_graph=show_graph)
    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()

    thz.plot_current(dataset)
    thz.global_truncate(dataset)

    thz.window_time(dataset, config={"window": {"type": "hann", "length": 0.3}}, show_graph=show_graph)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}}, show_graph=show_graph)
    thz.fft_spectrum(dataset)
    thz.trusted_band_mask(dataset, config={"mask": {"snr_thresh_db": 2,
                                                 "tail_fraction": 0.25,
                                                 "min_contiguous_bins": 3}})
    thz.transfer_function(dataset, config={"transfer": {"apply_snr_mask": True}}, ref_type='reference')
    # thz.time_shift_slider(dataset, shift_range_ps=(-0.1, 0.06), n_steps=100, sample="a-45_2", quantity='sigma')
    thz.plot_fft(dataset, freq_range=(0.3, 10), normalise=False, scale='')
# 
    # thz.phase_correction(dataset, source='fft')

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

    dataset.save_database()
    thz.result_viewer(dataset)
    thz.export_results(dataset)