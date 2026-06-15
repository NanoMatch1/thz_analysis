import os

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import analysis_tools as tools


# thisdir = os.path.dirname(os.path.abspath(__file__))

def plot_sigma(dataset, title=""):
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
    plt.title("Derived Conductivity {}".format(title))
    return 

def plot_nk(dataset, title=""):
    fig_nk, ax = plt.subplots()
    show_snr_mask = True
    cmap = plt.get_cmap('tab10')
    for index, (filename, data_obj) in enumerate(thz._sample_items(dataset)):
        freq = data_obj.processing_dict.get('fft_freq')
        n = data_obj.processing_dict.get('n')
        k = data_obj.processing_dict.get('k')
        if freq is None or n is None or k is None:
            continue
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
        thz._plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, n, mask, label="{} (n)".format(filename), color=cmap(index))
        thz._plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, k, mask, label="{} (k)".format(filename), color=cmap(index), linestyle='dashed')
    
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive Index / Extinction Coefficient")
    plt.title("Derived Optical Constants {}".format(title))
    # plt.show()
    return


# dataset = DataSet("C:/Users/Samuel/matchbook")
# dataset.load_database (index=35)
# # filtered_files = [file for file in dataset.data.keys() if "reference" not in file.lower()]

# plot_sigma(dataset, title="non-self-referenced")
# plot_nk(dataset, title="non-self-referenced")

# --- Self-referenced versions --- 
dataset = DataSet("C:/Users/Samuel/matchbook")
dataset.load_database()
plot_sigma(dataset, title="self-referenced")
plot_nk(dataset, title="self-referenced")



plt.show()