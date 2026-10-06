"""What is the knife edge actually seeing? Strip decomposition of the unclipped and 5 mm-aperture scans.

The detected signal with the blade at x0 is the overlap of the THz field with the detection mode,
integrated over the UNBLOCKED part of the blade plane (Kirchhoff):  S(x0, t) = integral_{x > x0}
s(x, t) dx.  So the negative derivative -dS/dx0 is s(x, t): the pulse carried by the strip of beam at
x, as the detector sees it. Its arrival time tau(x) is the relative wavefront between the THz beam and
the detection mode across the blade plane, and its spectrum |s(x, f)| is the beam profile per
frequency. If the "weird diffraction" is a wavefront mismatch, tau(x) is smooth (tilt = linear,
defocus = quadratic). If it is an edge effect, the strips nearest the blade misbehave instead.

Also checks the confounds: drift (position order == time order), the purge still settling, and the
positions re-measured out of order.

Data: ~/data/data_sync/diagnostics/2026-10-06_knife-edge-test{,_5mm-aperture}. Run from repo root.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diagnostics import knife_edge  # noqa: E402

DATA_ROOT = os.path.expanduser("~/data/data_sync/diagnostics")
SCANS = {
    "unclipped": os.path.join(DATA_ROOT, "2026-10-06_knife-edge-test"),
    "5 mm aperture": os.path.join(DATA_ROOT, "2026-10-06_knife-edge-test_5mm-aperture"),
}
FREQUENCIES_THZ = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
OUTPUT_DIRECTORY = os.environ.get("KNIFE_EDGE_OUTPUT",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))


def strip_pulses(table):
    """-dS/dx at strip midpoints: (midpoints_mm, strip traces (n-1, n_t), strip spectra)."""
    order = np.argsort(table.positions_mm)
    positions = table.positions_mm[order]
    traces = table.transformed_traces()[order]
    spectra = table.mean_spectra[order]
    step = np.diff(positions)[:, None]
    return (0.5 * (positions[1:] + positions[:-1]), -np.diff(traces, axis=0) / step,
            -np.diff(spectra, axis=0) / step)


def arrival_time_ps(time_ps, traces):
    """Envelope-centroid arrival time per trace (|analytic signal|^2 weighted), robust to shape."""
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(traces, axis=1)) ** 2
    return (envelope * time_ps).sum(axis=1) / envelope.sum(axis=1)


def group_delay_ps(frequencies_thz, spectra, reference, band_thz):
    """Delay of each spectrum relative to ``reference`` from the phase slope over a band."""
    band = (frequencies_thz >= band_thz[0]) & (frequencies_thz <= band_thz[1])
    delays = []
    for spectrum in spectra:
        phase = np.unwrap(np.angle(spectrum[band] / reference[band]))
        slope = np.polyfit(frequencies_thz[band], phase, 1)[0]
        delays.append(-slope / (2 * np.pi))
    return np.array(delays)


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    tables, fits = {}, {}
    for label, directory in SCANS.items():
        files, _ = knife_edge.load_knife_edge_files(directory,
                                                    exclude_filename_substrings=("purging",))
        tables[label] = knife_edge.build_knife_edge_table(files, FREQUENCIES_THZ)
        fits[label] = knife_edge.fit_all_frequencies(tables[label])
        print(f"\n===== {label} =====")
        knife_edge.print_fits(fits[label])

    # ---- open-beam comparison: what does the aperture remove? ----
    open_unclipped = tables["unclipped"].mean_spectra[0]
    open_aperture = tables["5 mm aperture"].mean_spectra[0]
    grid = tables["unclipped"].spectrum_frequencies_thz
    print("\nOpen-beam (0 mm) |E| ratio, aperture / unclipped:")
    for frequency in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        index = int(np.argmin(np.abs(grid - frequency)))
        print(f"  {grid[index]:.2f} THz  {abs(open_aperture[index]) / abs(open_unclipped[index]):.3f}")

    figure, axes = plt.subplots(3, 2, figsize=(12, 12))
    for column, (label, table) in enumerate(tables.items()):
        midpoints, strip_traces, strip_spectra = strip_pulses(table)
        frequencies = table.spectrum_frequencies_thz

        # time-domain strip image
        axis = axes[0, column]
        limit = np.abs(strip_traces).max()
        axis.imshow(strip_traces, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                    extent=(table.time_ps[0], table.time_ps[-1], midpoints[-1] + 0.5,
                            midpoints[0] - 0.5))
        arrival = arrival_time_ps(table.time_ps, strip_traces)
        strength = np.sqrt((strip_traces ** 2).sum(axis=1))
        strong = strength > 0.15 * strength.max()
        axis.plot(arrival[strong], midpoints[strong], "k.-", label="envelope centroid")
        axis.set(xlim=(101.5, 105.5), xlabel="time (ps)", ylabel="strip position (mm)",
                 title=f"{label}: pulse carried by each 1 mm strip (-dS/dx)")
        axis.legend(fontsize=8)

        # strip profile per frequency (the beam profile the detector sees)
        axis = axes[1, column]
        for frequency in FREQUENCIES_THZ:
            index = int(np.argmin(np.abs(frequencies - frequency)))
            profile = np.abs(strip_spectra[:, index])
            axis.plot(midpoints, profile / profile.max(), "o-", markersize=3,
                      label=f"{frequencies[index]:.2f} THz")
        axis.set(xlabel="strip position (mm)", ylabel="|s(x, f)| normalised",
                 title=f"{label}: beam profile per frequency")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)

        # strip delay vs position, two bands
        axis = axes[2, column]
        reference = strip_spectra[int(np.argmax(strength))]
        for band in ((0.4, 1.0), (1.0, 2.0)):
            delays = group_delay_ps(frequencies, strip_spectra, reference, band)
            axis.plot(midpoints[strong], 1e3 * delays[strong], "o-", label=f"{band[0]}-{band[1]} THz")
        axis.set(xlabel="strip position (mm)", ylabel="group delay vs strongest strip (fs)",
                 title=f"{label}: arrival time across the beam")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

        print(f"\n{label}: strip arrival (envelope centroid) vs position")
        for position, time, keep in zip(midpoints, arrival, strong):
            if keep:
                print(f"  x={position:5.1f} mm  t={time:.3f} ps")
        slope = np.polyfit(midpoints[strong], arrival[strong], 1)[0]
        print(f"  linear slope {1e3 * slope:.0f} fs/mm")

    figure.tight_layout()
    path = os.path.join(OUTPUT_DIRECTORY, "knife_edge_strip_analysis.png")
    figure.savefig(path, dpi=110)
    print(f"\nfigure: {path}")

    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    axis.semilogy(grid, np.abs(open_unclipped), label="unclipped, 0 mm")
    axis.semilogy(grid, np.abs(open_aperture), label="5 mm aperture, 0 mm")
    axis.semilogy(grid, np.abs(open_aperture) / np.abs(open_unclipped) * np.abs(open_unclipped).max(),
                  "k--", label="ratio (scaled)")
    axis.set(xlim=(0, 4), xlabel="frequency (THz)", ylabel="|E|", title="Open beam")
    axis.legend()
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIRECTORY, "knife_edge_open_beam.png"), dpi=110)

    figure = knife_edge.plot_knife_edge(tables["5 mm aperture"], fits["5 mm aperture"])
    figure.savefig(os.path.join(OUTPUT_DIRECTORY, "knife_edge_aperture_curves.png"), dpi=110)
    figure = knife_edge.plot_knife_edge(tables["unclipped"], fits["unclipped"])
    figure.savefig(os.path.join(OUTPUT_DIRECTORY, "knife_edge_unclipped_curves.png"), dpi=110)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Test: is the delay gradient the WHOLE story? Remove it and re-integrate.
# ---------------------------------------------------------------------------

def realigned_knife_edge(table, delay_slope_ps_per_mm):
    """|S(x0, f)| rebuilt from the strips after removing a linear delay across the beam.

    Returns (positions_mm, raw |S|, realigned |S|) on the measured grid; S at the last position is
    taken as the floor and strips are summed from the far side, as the blade sees them.
    """
    midpoints, _, strip_spectra = strip_pulses(table)
    frequencies_hz = table.spectrum_frequencies_thz * 1e12
    step = np.diff(np.sort(table.positions_mm))[:, None]
    # strip spectra carry exp(-2 pi i f tau(x)); multiplying by exp(+2 pi i f tau) removes it
    phase_removal = np.exp(2j * np.pi * frequencies_hz[None, :]
                           * delay_slope_ps_per_mm * 1e-12 * midpoints[:, None])
    order = np.argsort(table.positions_mm)
    floor = table.mean_spectra[order][-1]
    raw = floor + np.cumsum((strip_spectra * step)[::-1], axis=0)[::-1]
    realigned = floor + np.cumsum((strip_spectra * phase_removal * step)[::-1], axis=0)[::-1]
    positions = table.positions_mm[order][:-1]
    return positions, np.abs(raw), np.abs(realigned)


def realignment_report():
    for label, directory in SCANS.items():
        files, _ = knife_edge.load_knife_edge_files(directory,
                                                    exclude_filename_substrings=("purging",))
        table = knife_edge.build_knife_edge_table(files, FREQUENCIES_THZ)
        midpoints, strip_traces, strip_spectra = strip_pulses(table)
        strength = np.sqrt((strip_traces ** 2).sum(axis=1))
        strong = strength > 0.15 * strength.max()
        reference = strip_spectra[int(np.argmax(strength))]
        delays = group_delay_ps(table.spectrum_frequencies_thz, strip_spectra, reference,
                                (1.0, 2.0))
        slope = np.polyfit(midpoints[strong], delays[strong], 1)[0]
        positions, raw, realigned = realigned_knife_edge(table, slope)
        grid = table.spectrum_frequencies_thz
        print(f"\n{label}: delay slope (1-2 THz group delay) {1e3 * slope:.1f} fs/mm "
              f"= {np.degrees(np.arctan(slope * 1e-12 * 2.998e8 * 1e3)):.2f} deg wavefront tilt")
        print("  full-beam |S(first position)| gain if the gradient were removed:")
        for frequency in (0.5, 1.0, 1.5, 2.0, 2.5):
            index = int(np.argmin(np.abs(grid - frequency)))
            print(f"    {grid[index]:.2f} THz  x{realigned[0, index] / raw[0, index]:.2f}")
        print("  monotonic after realignment? (largest rise going INTO the beam, % of max)")
        for frequency in FREQUENCIES_THZ:
            index = int(np.argmin(np.abs(grid - frequency)))
            for name, curve in (("raw", raw[:, index]), ("realigned", realigned[:, index])):
                rise = np.max(np.diff(curve)) / curve.max() * 100
                print(f"    {grid[index]:.2f} THz {name:9s} {rise:5.1f}%", end="")
            print()


if __name__ == "__main__":
    realignment_report()
