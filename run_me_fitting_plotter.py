
"""Visualise THz-Core Drude fit results from .txt export files.

Each file has a `#`-prefixed header block (model info + fitted parameters)
followed by tab-delimited columns:
    freq_Hz  Re_data  Im_data  Re_fit  Im_fit  Resid_Re  Resid_Im
"""

from __future__ import annotations

import os
import re
import numpy as np
import matplotlib.pyplot as plt

_HZ_TO_THZ = 1e-12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FitResult:
    """Container for a single fit-result file."""
    def __init__(self, name, meta, freq_hz, re_data, im_data, re_fit, im_fit, resid_re, resid_im):
        self.name = name
        self.meta = meta          # dict of header metadata
        self.freq_hz = freq_hz
        self.re_data = re_data
        self.im_data = im_data
        self.re_fit  = re_fit
        self.im_fit  = im_fit
        self.resid_re = resid_re
        self.resid_im = resid_im

    @property
    def freq_thz(self):
        return self.freq_hz * _HZ_TO_THZ


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def loader(filepath: str) -> FitResult:
    """Parse a THz-Core fit-result .txt file into a FitResult object."""
    meta = {}
    data_lines = []

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.startswith("#"):
                if line.strip():
                    data_lines.append(line)
                continue

            content = line.lstrip("# ").strip()

            # Key : value pairs
            match = re.match(r"^([\w\s²χ]+?)\s*:\s*(.+)$", content)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                meta[key] = value

            # Indented parameters:  name = value ± err  [unit]
            param_match = re.match(
                r"^\s+(\w+)\s*=\s*([^\s]+)\s*.*\[(.+)\]", content
            )
            if param_match:
                param_name = param_match.group(1)
                param_value = float(param_match.group(3).split()[0]) if False else param_match.group(2)
                meta.setdefault("params", {})[param_name] = param_match.group(2)

    arr = np.loadtxt(data_lines)
    name = os.path.splitext(os.path.basename(filepath))[0]

    return FitResult(
        name=name,
        meta=meta,
        freq_hz=arr[:, 0],
        re_data=arr[:, 1],
        im_data=arr[:, 2],
        re_fit=arr[:, 3],
        im_fit=arr[:, 4],
        resid_re=arr[:, 5],
        resid_im=arr[:, 6],
    )


def load_directory(directory: str) -> list[FitResult]:
    """Load all .txt fit-result files in a directory."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".txt"):
            results.append(loader(os.path.join(directory, fname)))
    return results


# ---------------------------------------------------------------------------
# Plotting helpers (style consistent with open_session.py)
# ---------------------------------------------------------------------------

def _label_with_quality(result: FitResult, suffix: str) -> str:
    r2 = result.meta.get("R", result.meta.get("R²", "?"))
    return f"{result.name} {suffix} (R²={r2})"


def plot_fit_results(fit_results: list[FitResult], title: str = "") -> None:
    """Plot raw data (scatter) and Drude fit (line) for each result on one axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    for index, result in enumerate(fit_results):
        color = cmap(index)
        freq = result.freq_thz
        label_base = _label_with_quality(result, "")

        # Measured data — scatter, matching open_session.py style
        ax.scatter(freq, result.re_data, s=12, marker="o", color=color,
                   alpha=0.7, label=f"{label_base} data (Re)")
        ax.scatter(freq, result.im_data, s=12, marker="x", color=color,
                   alpha=0.7, label=f"{label_base} data (Im)")

        # Fitted curve — solid line for real, dashed for imaginary
        ax.plot(freq, result.re_fit, color=color, linestyle="solid",  linewidth=1.5)
        ax.plot(freq, result.im_fit, color=color, linestyle="dashed", linewidth=1.5)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Conductivity (S/m)")
    ax.set_title(f"Drude Fit Results{' — ' + title if title else ''}")
    ax.legend(fontsize=7)
    fig.tight_layout()


def plot_residuals(fit_results: list[FitResult], title: str = "") -> None:
    """Plot fit residuals for each result."""
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.get_cmap("tab10")

    for index, result in enumerate(fit_results):
        color = cmap(index)
        freq = result.freq_thz
        ax.plot(freq, result.resid_re, color=color, linestyle="solid",
                linewidth=1.2, label=f"{result.name} Re")
        ax.plot(freq, result.resid_im, color=color, linestyle="dashed",
                linewidth=1.2, label=f"{result.name} Im")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Residual (S/m)")
    ax.set_title(f"Fit Residuals{' — ' + title if title else ''}")
    ax.legend(fontsize=7)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
config = {
    'guideline': False,
    'normalise': True,
    'cutoff_freq_thz': 4.0,  # Example cutoff frequency for masking
}

file_dir = r'C:\Users\Samuel\Data\THz\calibration\silicon\2026-07-03_silicon_standards\fitted'

if __name__ == "__main__":
    fit_results = load_directory(file_dir)
    print(f"Loaded {len(fit_results)} fit result(s):")
    for result in fit_results:
        print(f"  {result.name}  —  R²={result.meta.get('R', result.meta.get('R²', '?'))}")

    plot_fit_results(fit_results, title="Silicon standards")
    plot_residuals(fit_results, title="Silicon standards")
    plt.show()
