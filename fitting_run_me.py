# Demo: synthetic Drude conductivity data
import os

import numpy as np

from thz_core.build.lib.thz_core.fit_gui import FitGUI
from thz_core.thz_core.fitting._models import drude_conductivity
import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools

def load_dataset(index):
    dataset = DataSet("C:/Users/Samuel/matchbook")
    dataset.load_database(index=index)

    data_dict = {}

    for filename, data_obj in thz._sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        sigma = data_obj.processing_dict.get('sigma')

        # sigma.real[0] = 1.8e6  # Set DC conductivity to MS
        # sigma.imag[0] = 0.0  # Set DC conductivity to purely real


        if freq is None or sigma is None:
            continue
        real = sigma.real
        imag = sigma.imag
        mask = data_obj.processing_dict.get('transfer_mask')
        # export data
        exportDir = r'C:\Users\Samuel\Data\THz\Sam\Analysis\exports'
        data = np.column_stack((freq, real, imag))
        np.savetxt(os.path.join(exportDir, f"{filename}_sigma.csv"), data, delimiter=",", header="Frequency (Hz), Real(Sigma), Imag(Sigma)", comments="")

        print(f"\n=== {filename} ===")
        gui = FitGUI(freq, sigma, mask=mask, initial_model="drude_conductivity")
        result = gui.run()
        if result:
            print(f"\nFit result for {filename}:")
            print("\n".join(result.summary_lines()))
            data_dict[filename] = {
                "freq": freq,
                "sigma": sigma,
                "fit_result": result
            }

        data_obj.processing_dict["fit_result"] = result


    # dataset.save_state()  # Save the fit results to a .state file for later use
    # dataset.save_database()  # Save the updated processing_dict to the database

load_dataset(index=40)  # Change the index to load a different dataset
    
# freq = np.linspace(0.1e12, 5e12, 300)
# omega = 2 * np.pi * freq
# sigma_dc, tau = 800.0, 2e-13
# rng = np.random.default_rng(0)
# noise = rng.normal(0, 5, len(freq)) + 1j * rng.normal(0, 5, len(freq))
# data = drude_conductivity(omega, sigma_dc, tau) + noise
# mask = (freq >= 0.2e12) & (freq <= 3e12)

# gui = FitGUI(freq, data, mask=mask, initial_model="drude_conductivity")
# result = gui.run()
# if result:
#     print("\nFit result:")
#     print("\n".join(result.summary_lines()))
    