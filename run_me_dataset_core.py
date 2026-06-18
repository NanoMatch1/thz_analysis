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

def purging_analysis(dataset):
    """Analyzes the effect of purging on the THz signal by comparing scans taken at different times since purging start. Computes a deviation metric based on the standard deviation of the difference between consecutive scans and plots it over time to visualize any trends.
    
    Final output is the decay exponential fit parameters for each file, which can be used to compare the purging dynamics across different samples or conditions."""

    from scipy.optimize import curve_fit
    def exp_decay(t, a, b):
        return a * np.exp(-b * t)

    dev_dict = {}
    decays = []
    for filename, data_obj in dataset.data.items():
        # breakpoint()
        dataX = data_obj.raw_data[:, 0]
        # plt.plot()
        dev_list = []
        mask = [True if (x > 154) and (x < 158) else False for x in dataX]
        timestamps = data_obj.resolve_timestamps()
        normalised_time = np.array([(t - timestamps[0]).total_seconds() for t in timestamps]) if timestamps is not None else None
        for index in range(data_obj.raw_data.shape[1]):
            timestamp = normalised_time[index-1] if index > 0 else None
            if index == 0:
                last_trace = np.zeros_like(dataX[mask])
                continue
            dataY = data_obj.raw_data[mask, index]
            difference = dataY - last_trace
            deviation = np.std(difference) # gen deviation metric
            last_trace = np.copy(dataY)
            dev_list.append([timestamp, deviation])
        dev_dict[filename] = dev_list

    for filename, dev_list in dev_dict.items():
        data = np.array(dev_list)
        plt.scatter(data[:, 0], data[:, 1], label=filename)
        
        # Fit exponential decay        
        try:
            popt, _ = curve_fit(exp_decay, data[:, 0], data[:, 1], p0=[data[0, 1], 0.1], maxfev=10000)
            t_fit = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            y_fit = exp_decay(t_fit, *popt)
            plt.plot(t_fit, y_fit, label=f'{filename} (fit: a={popt[0]:.2e}, b={popt[1]:.2e})', linestyle='--')

            # print(f"{filename}: Fitted exponential decay parameters: a = {popt[0]:.2e}, b = {popt[1]:.2e}")
            decays = decays + [(filename, popt[0], popt[1])]
        except:
            pass

    sorted_decays = sorted(decays, key=lambda x: x[2])[::-1]  # sort by decay rate (b)
    print("Purging analysis complete. Decay parameters (sorted by decay rate, fastest first):")
    for filename, a, b in sorted_decays:
        print(f"{filename}: a = {a:.2e}, b = {b:.2e}")
        
    plt.legend()
    plt.title("Deviation Metric Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Standard Deviation of Difference from Previous Trace")
    plt.show()



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
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\A\segmented\second_reflection"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\H2O_timing tests"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D\segmented\second_reflection"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-16\A"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\testing\cryostat_windows"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\testing\silicon"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-11_CNT-paper"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-16_reflection_testing\export"
    # fileDir = r"C:\Users\Samuel\Data\THz\diagnostics\2026-06-10_humidity and purge\2026-06-09_CNT-paper\TESTING"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\B\segmented\second_reflection"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-12\B\segmented\second_reflection"



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
    
    def window_characterisation(dataset):
        '''Performs window characterisation using the first and second reflection segments of a reference scan to extract the true complex refractive index to improve accuracy of the subsequent inversion.'''

        references = dataset.data.references
        if len(references) == 0:
            print("No reference files found for window characterisation.")
            return
        if len(references) > 1:
            print("Multiple reference files found. Using the first one for window characterisation.")
        reference_filename = next(iter(references.keys()))
        print(f"Using reference file: {reference_filename}")

        # result = thz.characterise_window(
        first_refl = os.path.join(fileDir, "first_reflection", reference_filename)
        second_refl = os.path.join(fileDir, "second_reflection", reference_filename)
        result = thz.characterise_window(first_refl, second_refl, thickness_m=0.9e-3, theta_deg=45.0, band_thz=(0.2, 3.5), show_graph=True)

        n_window_freq = result['n'] - 1j * result['k'] # Build the complex refractive index of the window material from the characterisation result.
        return n_window_freq

    ### --- Config Setup ---
   
    def preprocess(fileDir, show_graph=True):
        dataset = DataSet(fileDir)
        dataset.load_all_data(case_insensitive=True, explicit_dir=True)
        # dataset.plot_current()
        # --- For reflection data ---
        dataset.group_files(keywords=['type'])
        thz.subtract_baseline(dataset, show_graph=show_graph)
        # --- THz-TDS processing pipeline ---
        # --- initial pre-processing steps ---

        # --- alignment step, different protocols
        thz.align_to_reference(dataset, ref_type="reference", roi=(152, 156), subsample_correction=False, show_graph=show_graph)
        # dataset.plot_current()

        thz.normalise(dataset, config={"bounds": (152, 156)}, show_graph=show_graph)
        thz.segment_reflections(dataset, show_graph=True, segments={'first_reflection': (152, 159), 'second_reflection': (162.2, 168)})
        print("Pre-processing complete.")
        sys.exit()

        # dataset.save_state()

# 15 mins for 6-aligned


    ### --- Acquisition editing ---
    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)
    ### --- Pre-processing and segmentation ---
    # --- Preprocess reflections - uncomment to segment and normalise time traces. Comment and re-run with segmented folder to continue analysis
    # preprocess(fileDir)
    # --- Main analysis 
    dataset = DataSet(fileDir)
    show_graph = True

    fresh_load = False
    if fresh_load:

        dataset.load_all_data(case_insensitive=True, explicit_dir=True)
        dataset.plot_current()
        # dataset.plot_current()
        thz.subtract_baseline(dataset, show_graph=show_graph)
        # thz.build_full_trace_reflection()
        

        # ---


        dataset.group_files(keywords=['type', 'seri'])
        dataset.grouping.show_matches()


        # thz.plot_current(dataset)
        # thz.global_truncate(dataset)
        dataset.plot_current(title="Current Dataset")
        # thz.centering_manual(dataset, show_graph=True)#, auto_range_ps=(163,167), recalibrate=False)
        # --- Pre-window centering: extend traces backward so the pulse sits at the
        # temporal midpoint, giving the Tukey window symmetric taper regions.
        centering_config = {'centering': {'peak_mode': 'manual', 'taper_ps': 1}}
        thz.center_pulse(dataset, config=centering_config, show_graph=show_graph)
        # thz.center_first_reflection_pulse(dataset, config=centering_config, show_graph=show_graph)
        thz.window_time(dataset, config={"window": {"type": "tukey", "alpha": 1}}, show_graph=show_graph)
        # thz.zero_pad(dataset, config={"pad": {"extend_factor": 1.0}}, show_graph=show_graph)
        thz.align_to_common_time_axis(dataset, show_graph=show_graph)
        thz.fft_spectrum(dataset)
        # The SNR mask is built inside transfer_function (it needs H), so the mask
        # thresholds must travel with it — a standalone trusted_band_mask call before
        # this point is a no-op (transfer_H not computed yet) and silently drops them.
        # self_reference: front-pulse drift correction from the sibling
        # first_reflection segment folder (ANALYSIS_NOTES §11). Set False to compare
        # against the conventional bare-window reference.
        dataset.save_state()
    else:
        dataset.load_state()
        # dataset.plot_current(title="Current Dataset")

    thz.transfer_function(
        dataset,
        config={
            "transfer": {"apply_snr_mask": True, "self_reference": False},
            "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        },
        ref_type='reference',
    )
    # thz.remove_phase_offset(dataset, config={"phase_offset": {"band_thz": (0.3, 2.0)}}, show_graph=show_graph)
    # TODO(revisit): invert_nk uses anchor_phase_origin=True by default (removes the
    # whole-cycle 2pi wrap that droops n at low f for thick samples). Left ON for now —
    # check this is still desired/correct once more transmission + reflection data is in.
    # thz.time_shift_slider(dataset, shift_range_ps=(-0.1, 0.06), n_steps=100, sample="a-45_2", quantity='sigma')

    # thz.plot_fft(dataset, freq_range=(0.0, 10), normalise=False, scale='')
# 
    # thz.phase_correction(dataset, source='transfer')

    # thz.invert_nk(dataset, thickness_m=2.08e-3)
    thz.invert_nk(dataset, thickness_m=350e-6)
    thz.derive_eps_sigma(dataset)

    # n_window_freq = window_characterisation(dataset)
    # thz.invert_nk_reflection(
    #     dataset,
    #     geometry="window",
    #     theta_deg=45,
    #     polarization='s',
    #     # n_window=1.964,
    #     n_window=1.964,
    # )
    # # thz.invert_nk_reflection(
    # #     dataset,
    # #     geometry="gold",
    # #     theta_deg=45,
    # #     polarization='s',
    # #     # n_window=1.95,
    # # )

    dataset.save_database()
    thz.result_viewer(dataset)
    # thz.export_results(dataset)
    def plot_sigma(dataset):
        fig_sigma, ax = plt.subplots()
        show_snr_mask = True
        cmap = plt.get_cmap('tab10')
        for index, (filename, data_obj) in enumerate(thz._sample_items(dataset)):
            freq = data_obj.processing_dict.get('fft_freq')
            # n = data_obj.processing_dict.get('n')
            # k = data_obj.processing_dict.get('k')
            sigma = data_obj.processing_dict.get('sigma')
            real = sigma.real
            imag = sigma.imag
            if freq is None or real is None or imag is None:
                continue
            mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
            thz._plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, real, mask, label="{} (real)".format(filename), color=cmap(index))
            thz._plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, imag, mask, label="{} (imag)".format(filename), color=cmap(index), linestyle='dashed')
        
        plt.legend()
        plt.xlabel("Frequency (THz)")
        # plt.ylabel("Refractive Index / Extinction Coefficient")
        plt.ylabel("Conductivity (S/m)")
        plt.title("Derived Conductivity")
        return 


    plot_sigma(dataset)
    plt.show()