"""Reopen a saved analysis session bundle (.thzbundle) — two ways.

A bundle is written by any run_me_*.py with config['general']['save_session'] set (or by calling
``session_bundle.save_session(dataset, path)`` directly). It contains:

    recipe.json    editable pipeline spec (source dir + config + ordered steps + git SHA)
    snapshot.pkl   the processed results (instant display, no recompute)
    report.md      human-readable processing report

Usage:
    python open_session.py <path/to/run.thzbundle>            # load snapshot + launch viewer
    python open_session.py <path/to/run.thzbundle> --replay   # recompute from raw data
"""

from __future__ import annotations

import sys
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import session_bundle, pipeline_registry


def extract_sigma(dataset):
    """Extract sigma from the dataset and return a dictionary of results."""
    data_dict = {}
    for filename, data_obj in thz._sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        sigma = data_obj.processing_dict.get('sigma')
        if freq is None or sigma is None:
            continue
        data_dict[filename] = {
            "freq": freq,
            "sigma": sigma
        }
    return data_dict

def generate_channel(dataX, dataY, depth=0.2):
    '''Generate a channel around the dataY curve for visualization tracing.'''
    upper = dataY + dataY * depth
    lower = dataY - dataY * depth
    return upper, lower

def _plot_with_snr_mask(ax, freq_thz, dataY, mask, label=None, color=None, linestyle='solid', marker='o', config={}):
    """Plot dataY vs freq_thz, optionally masking out low-SNR points."""
    guideline = config.get('guideline', False)
    normalise = config.get('normalise', False)
    if normalise:
        dataY = dataY / np.max(np.abs(dataY))  # Normalize to the maximum absolute value
    if mask is not None:
        ax.scatter(freq_thz[mask], dataY[mask], label=label, s=10, marker=marker, color=color, alpha=0.8) # high-SNR points
        ax.scatter(freq_thz[~mask], dataY[~mask], color=color, marker='.', alpha=0.1, s=1, label=None)  # low-SNR points, faint
    else:
        ax.scatter(freq_thz, dataY, label=label, color=color, marker=marker, alpha=0.8, s=10)
    if guideline:
        upper, lower = generate_channel(freq_thz, dataY)
        ax.fill_between(freq_thz, lower, upper, color=color, alpha=0.1)  # guideline for reference
    
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

        _plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, real, mask, label="{} (real)".format(filename), color=cmap(index))
        _plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, imag, mask, label="{} (imag)".format(filename), color=cmap(index), marker='x')
    ax.set_xlabel("Frequency (THz)")
    # plt.ylabel("Refractive Index / Extinction Coefficient")
    ax.set_ylabel("Conductivity (S/m)")
    ax.set_title("Derived Conductivity {}".format(title))
    ax.legend()
    return 

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    bundle_dir = sys.argv[1]
    replay = "--replay" in sys.argv[2:]

    if replay:
        # Recompute headlessly from the raw data using the recorded recipe. Edit recipe.json
        # first (or pass override_config to replay_recipe) to re-run with tweaks.
        recipe = session_bundle.read_recipe(bundle_dir)
        print(f"Replaying {len(recipe.get('steps', []))} steps from {bundle_dir} ...")
        dataset = pipeline_registry.replay_recipe(recipe)
    else:
        # Instant reload of the saved results — no recompute.
        dataset = session_bundle.load_session(bundle_dir)

    thz.result_viewer(dataset)

def run_me(data_dir=None):
    if not data_dir:
        raise ValueError("Please provide a path to the .thzbundle file.")
    
    basename = os.path.basename(data_dir)
    bundle_dir = os.path.join(data_dir, "{}.thzbundle".format(basename))
    
    recipe = session_bundle.read_recipe(bundle_dir)
    dataset = session_bundle.load_session(bundle_dir)

    plot_sigma(dataset, title="from session bundle")

    sigma_reflection = extract_sigma(dataset)

    return dataset, sigma_reflection


if __name__ == "__main__":
    file_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon\export'

    dataset_refl, sigma_refl = run_me(file_dir)

    # thz.launch_results_viewer(dataset_refl)
    # plt.show()

    file_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans\ntype'
    dataset_trans, sigma_trans = run_me(file_dir)

    thz.launch_results_viewer(dataset_trans)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab10')

    config = {
        'guideline': False,
        'normalise': False,
        'cutoff_freq_thz': 4.0,  # Example cutoff frequency for masking
    }

    for index, (filename, data) in enumerate(sigma_refl.items()):
        # breakpoint()
        mask = np.array([freq * thz._HZ_TO_THZ < config['cutoff_freq_thz'] for freq in data["freq"]])  # Example mask: frequencies below 4 THz
        # breakpoint()
        _plot_with_snr_mask(ax, data["freq"] * thz._HZ_TO_THZ, data["sigma"].real, mask=mask, label="{} (refl)".format(filename), color=cmap(0), marker='o', config=config)
        _plot_with_snr_mask(ax, data["freq"] * thz._HZ_TO_THZ, data["sigma"].imag, mask=mask, label="{} (refl)".format(filename), color=cmap(0), marker='x', config=config)

    for filename, data in sigma_trans.items():
        mask = np.array([freq * thz._HZ_TO_THZ < config['cutoff_freq_thz'] for freq in data["freq"]])  # Example mask: frequencies below 4 THz
        _plot_with_snr_mask(ax, data["freq"] * thz._HZ_TO_THZ, data["sigma"].real, mask=mask, label="{} (trans)".format(filename), color=cmap(1), marker='o', config=config)
        _plot_with_snr_mask(ax, data["freq"] * thz._HZ_TO_THZ, data["sigma"].imag, mask=mask, label="{} (trans)".format(filename), color=cmap(1), marker='x', config=config)

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Conductivity (S/m)")
    ax.set_title("Reflection vs Transmission — Derived Conductivity")
    ax.legend()

    plt.show()
    # main()