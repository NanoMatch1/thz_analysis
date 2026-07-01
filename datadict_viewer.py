import os

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
import pickle

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools


def load_data_dict(data_dir, config=None):
    """Load all .dic files via pickle protocol returning a data dictionary."""

    data_dict = {}

    dic_files = [f for f in os.listdir(data_dir) if f.endswith('.dic')]
    for dic_file in dic_files:
        with open(os.path.join(data_dir, dic_file), 'rb') as f:
            new_data = pickle.load(f)
        data_dict[dic_file.replace('.dic', '')] = new_data      

    return data_dict

def filter_by_string(data_dict, substring):
    """Filter a data dictionary by matching the substring in the keys."""
    return {k: v for k, v in data_dict.items() if substring in k}


if __name__ == "__main__":
    # Example usage
    data_dir = r'C:\Users\Samuel\Data\THz\diagnostics\2026-07-01_cnt-rotation'
    config = None  # or provide a configuration dictionary if needed
    data_dict = load_data_dict(data_dir, config=config)
    print(f"Loaded {len(data_dict)} data entries from {data_dir}.")

    cnt_data = data_dict.get('cnt_noalign')
    # breakpoint()

    cnt180 = filter_by_string(cnt_data, 'CNT-0-180')
    cnt90 = filter_by_string(cnt_data, 'CNT-0-90')
    cnt0 = filter_by_string(cnt_data, 'CNT-0-0')
    cnt270 = filter_by_string(cnt_data, 'CNT-0-270')
    reference = filter_by_string(cnt_data, 'reference')

    cntS = {}
    cntS.update(cnt0)
    cntS.update(cnt180)

    cntP = {}
    cntP.update(cnt90)
    cntP.update(cnt270)

    # blues = plt.get_cmap('Blues')
    # greens = plt.get_cmap('Greens')
    # reds = plt.get_cmap('Reds')
    # breakpoint()

    def plot_comparison(data_dict, reference, series_label=""):
        """Plot the FFT spectra and transfer functions for the given CNT data and reference."""
        reference_data = list(reference.values())[0]
        refX = reference_data['fft_freq'] * thz._HZ_TO_THZ
        refY = np.abs(reference_data['fft_spectrum'])

        fig, ax = plt.subplots(1, 2)

        for filename, data_dict in data_dict.items():
            print(filename)
            color = 'tab:blue' if 'CNT-0-0' in filename or 'CNT-0-90' in filename else 'tab:orange'
            dataX = data_dict['fft_freq'] * thz._HZ_TO_THZ
            dataY = np.abs(data_dict['fft_spectrum'])
            label = filename.split('_CNT-')[1].split('_')[0]
            # plt.plot(dataX, dataY, label=label, color=blues(0
            print(label)
            ax[0].plot(dataX, dataY, label=filename, color=color)

            diff = np.abs(dataY/refY)
            ax[1].plot(dataX, diff, label=filename, color=color)


        ax[0].plot(refX, refY, label='Reference', color='black', linestyle='--')
        ax[1].plot(refX, np.ones_like(refY), label='Reference', color='black', linestyle='--')

        # ax[0].set_yscale('log')

        ax[0].set_xlabel('Frequency (THz)')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_title('FFT Spectrum {}'.format(series_label))
        ax[0].legend()

        ax[1].set_xlabel('Frequency (THz)')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_title('Transfer Function {}'.format(series_label))
        ax[1].legend()

        plt.show()

# plot_comparison(cntS, reference, series_label="CNT-S")
plot_comparison(cntP, reference, series_label="CNT-P")