"""Dig into a saved analysis: pull its arrays out as plain dictionaries to process and plot.

``open_session.py`` is for *looking* at a bundle. This is for *working* with one: load it from
the catalogue, then get exactly the arrays the viewer would draw — already masked-ready, with
error bars — as nested dicts you can manipulate, combine across files, or plot however you like.

Every frequency-domain quantity comes from the quantity registry (``display.get_series``), so
anything registered there (n, k, sigma_real, transfer_phase, fft_mag, ...) is extractable by name
with no extra code. Time traces come from ``display.get_time_series``.

Shapes returned:
    extract_series(dataset, ["n", "k"])
        -> {"n": {filename: {"freq_hz", "freq_thz", "y", "error", "mask"}}, "k": {...}}
    display.get_time_series(dataset)
        -> {filename: {"time_s", "time_ps", "amplitude", "error"}}

``mask`` is the trusted-band / SNR mask (boolean, or None if the stage stored none) and
``error`` the Monte-Carlo or measured sigma (or None). For anything not in the registry, the
full per-file store is ``dataset.data.data_dict[filename].processing_dict``.

    python extract_session_data.py                # pick a bundle, list what's extractable
    python extract_session_data.py 49ad3cc4       # same, for a given id-prefix or series name

In a REPL / debugger:
    from extract_session_data import load_dataset, extract_series
    dataset = load_dataset("49ad3cc4")
    data = extract_series(dataset, ["n", "k", "sigma_real"])
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from dataset_core.adapters import display
from open_session import open_from_catalogue


def load_dataset(identifier: str | None = None):
    """Load a bundle's saved results by id-prefix / series name (picker if None)."""
    dataset = open_from_catalogue(identifier)
    if dataset is None:
        raise SystemExit("Nothing selected.")
    return dataset


def extract_series(dataset, quantities, *, files=None) -> dict:
    """``{quantity: {filename: arrays}}`` for each requested registered quantity.

    ``files`` selects items exactly as in ``display.plot_quantity`` (None = samples, plus
    references for reference-meaningful quantities; or a regex, list, or predicate).
    """
    return {quantity: display.get_series(dataset, quantity, files=files) for quantity in quantities}


def trusted(arrays: dict, key: str = "y"):
    """``(freq_thz, values)`` restricted to the trusted mask, for quick custom plotting."""
    mask = arrays["mask"]
    if mask is None:
        return arrays["freq_thz"], arrays[key]
    mask = np.asarray(mask, dtype=bool)
    return arrays["freq_thz"][mask], np.asarray(arrays[key])[mask]


def summarise(dataset) -> None:
    """Print what this dataset holds: files, extractable quantities, trusted ranges."""
    names = display.available_quantities(dataset)
    print(f"\nFiles ({len(dataset.data.data_dict)}):")
    for filename in dataset.data.data_dict:
        print(f"  {filename}")
    print(f"\nExtractable quantities: {', '.join(names)}")
    for quantity, per_file in extract_series(dataset, names).items():
        for filename, arrays in per_file.items():
            freq_thz, values = trusted(arrays)
            if len(freq_thz) == 0:
                continue
            error_note = "  +error" if arrays["error"] is not None else ""
            print(f"  {quantity:14} {filename:40} {freq_thz.min():5.2f}-{freq_thz.max():5.2f} THz  "
                  f"({len(freq_thz)} trusted points){error_note}")


def example_custom_plot(dataset) -> None:
    """A worked example: n and k of every sample on one axis, trusted band only, with errors."""
    data = extract_series(dataset, ["n", "k"])
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for filename, arrays in data["n"].items():
        freq_thz, n_values = trusted(arrays)
        line, = axis.plot(freq_thz, n_values, label=f"{filename} n")
        if arrays["error"] is not None:
            _, n_error = trusted(arrays, key="error")
            axis.fill_between(freq_thz, n_values - n_error, n_values + n_error,
                              color=line.get_color(), alpha=0.2)
        if filename in data["k"]:
            freq_thz, k_values = trusted(data["k"][filename])
            axis.plot(freq_thz, k_values, linestyle="--", color=line.get_color(), label=f"{filename} k")
    axis.set_xlabel("Frequency (THz)")
    axis.set_ylabel("n (solid), k (dashed)")
    axis.legend(fontsize=7)
    figure.tight_layout()


if __name__ == "__main__":
    positionals = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    dataset = load_dataset(positionals[0] if positionals else None)
    summarise(dataset)
    if "--plot" in sys.argv:
        example_custom_plot(dataset)
        plt.show()
