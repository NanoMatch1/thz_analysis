import os

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools

from pathlib import Path

def export_thz_data_as_acc(data_obj, dest_path):
    """Export a THzData object to an .acc file using acquisition_editor.save_acc."""
    data_dict = {
        "filename": Path(dest_path).name,
        "header": data_obj.headers,                              # first-scan header lines
        "scan_headers": [s.headers for s in data_obj.data_list], # per-scan headers
        "data": data_obj.raw_data,                               # (N_pts, 1+N_scans), time in ps
    }
    print(f"Exporting THzData to {dest_path}...")
    return acquisition_editor.save_acc(data_dict, dest_path)


def clone_and_export(data_obj, dest_path, time_shift_ps=0.0):
    """Clone a THzData object, apply a time shift, and export to an .acc file."""
    import copy
    synthetic = copy.deepcopy(data_obj)
    synthetic._data[:, 0] += time_shift_ps          # shift time
    synthetic.raw_data[:, 0] += time_shift_ps      # shift time
    synthetic._time_data += time_shift_ps          # shift time
    return export_thz_data_as_acc(synthetic, dest_path)
    
if __name__ == "__main__":
    import sys
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026_06_17_reflection_setup_large\holder"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\calibration_testing\2026-06-01_cryostat_windows"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\cone_tests"
    fileDir = r"C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-24_refl_testing\2025-06-25_silicon"
    # fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\aligned_references"
    
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
    # --- Main analysis ---
    dataset = DataSet(fileDir)
    show_graph = True

    fresh_load = True
    if fresh_load:

        dataset.load_all_data(case_insensitive=True, explicit_dir=True)
        # for filename, data_obj in dataset.data.items():
        #     if 'reference_gold_camera-align_back-face' in filename.lower():
        #         clone_and_export(data_obj, os.path.join(fileDir, "sample_synthetic_reference_+2.acc"), time_shift_ps=2.0)
        #         clone_and_export(data_obj, os.path.join(fileDir, "sample_synthetic_reference_-2.acc"), time_shift_ps=-2.0)

        dataset.group_files(keywords=['type'])
        dataset.grouping.show_matches()

        thz.subtract_baseline(dataset)
        thz.centering_manual(dataset, 
                            #  auto_range_ps=(163,167), 
                             recalibrate=False
                             )
        dataset.plot_current()
        # --- Pre-window centering: extend traces backward so the pulse sits at the
        # temporal midpoint, giving the Tukey window symmetric taper regions.
        thz.window_time(dataset, config={"window": {"type": "tukey", "alpha": 1}}, show_graph=show_graph)
        thz.zero_pad(dataset, config={"pad": {"extend_factor": 1.0}}, show_graph=show_graph)
        thz.fft_spectrum(dataset)
        dataset.save_state()
    else:
        dataset.load_state()

    # thz.phase_correction(dataset, source='fft')
    
    thz.transfer_function(
        dataset,
        config={
            "transfer": {"apply_snr_mask": True, "self_reference": False},
            "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        },
        ref_type='reference',
    )
    thz.plot_fft(dataset, freq_range=(0.0, 10), normalise=False, scale='')
    thz.invert_nk(dataset, thickness_m=300e-6)
    thz.derive_eps_sigma(dataset, config={"derive": {"eps_background": 1}})

    dataset.save_database()
    thz.result_viewer(dataset)