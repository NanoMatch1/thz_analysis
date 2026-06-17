import os

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools
    
if __name__ == "__main__":
    import sys
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026_06_17_reflection_setup_large\holder"
    
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


    ### --- Acquisition editing ---
    # import acquisition_editor
    # acquisition_editor.process_directory(fileDir)
    # --- Main analysis 
    dataset = DataSet(fileDir)
    show_graph = True

    fresh_load = True
    if fresh_load:

        dataset.load_all_data(case_insensitive=True, explicit_dir=True)
        dataset.group_files(keywords=['type'])
        dataset.grouping.show_matches()

        thz.subtract_baseline(dataset, show_graph=show_graph)
        thz.centering_manual(dataset, show_graph=True)#, auto_range_ps=(163,167), recalibrate=False)
        dataset.plot_current()
        # --- Pre-window centering: extend traces backward so the pulse sits at the
        # temporal midpoint, giving the Tukey window symmetric taper regions.
        thz.window_time(dataset, config={"window": {"type": "tukey", "alpha": 1}}, show_graph=show_graph)
        thz.zero_pad(dataset, config={"pad": {"extend_factor": 1.0}}, show_graph=show_graph)
        thz.fft_spectrum(dataset)
        dataset.save_state()
    else:
        dataset.load_state()

    thz.phase_correction(dataset, source='fft')
    
    thz.transfer_function(
        dataset,
        config={
            "transfer": {"apply_snr_mask": True, "self_reference": False},
            "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        },
        ref_type='reference',
    )
    thz.plot_fft(dataset, freq_range=(0.0, 10), normalise=False, scale='')
    thz.invert_nk(dataset, thickness_m=2.08e-3)
    thz.derive_eps_sigma(dataset)

    dataset.save_database()
    thz.result_viewer(dataset)