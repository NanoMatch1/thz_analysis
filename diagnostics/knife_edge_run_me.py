"""Knife-edge beam profile: spectral amplitude/intensity against blade position, per frequency.

One ``.acc`` per blade position, the position in the filename as ``..._<pos>mm...``. Prints the
table, fits an erf edge per frequency (beam radius vs frequency), writes a CSV of the flattened
columns and shows the figure. The physics -- why the fit is on |E| and not |E|^2 -- is in the
docstring of ``diagnostics/knife_edge.py``.

Run from the repo root:

    python diagnostics/knife_edge_run_me.py                    # the directory in config
    python diagnostics/knife_edge_run_me.py <directory>        # any other directory
    python diagnostics/knife_edge_run_me.py <directory> --no-graph
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics import knife_edge  # noqa: E402

DATA_ROOT = os.environ.get("THZ_DATA_ROOT", os.path.expanduser("~/data"))

config: dict = {
    "data_directory": os.path.join(DATA_ROOT, "data_sync", "diagnostics",
                                   "2026-10-06_knife-edge-test"),
    # Files whose name contains any of these are skipped (e.g. a repeat or a purge run taken at
    # one of the positions, which would otherwise appear as a second point there).
    "exclude_filename_substrings": ("purging",),

    # Frequencies to trace. The nearest FFT bin is used; the table prints the bin actually taken.
    # Bins closer together than the printed resolution (1 / record length) are not independent.
    "frequencies_thz": (0.3, 0.5, 0.75, 1.0, 1.5, 2.0),

    "transform": {
        # None = the whole record, unwindowed. A number = Tukey window of that half-width centred
        # on the mean pulse (one centre for every file, so a pulse that walks with the blade is
        # not re-centred file by file).
        "window_half_width_ps": None,
        "window_shape": "tukey",
        "taper_fraction": 0.5,
        "pad_factor": 4,
        "baseline_fraction": 0.1,     # leading share of each trace used as its DC baseline
    },

    "fit": {
        "enabled": True,
        # Below this share of the edge inside the measured range, the radius is flagged.
        "minimum_covered_fraction": 0.8,
        # Above this the erf does not describe the points to within their noise; flagged.
        "maximum_reduced_chi_square": 10.0,
    },

    "display": {
        "quantities": ("amplitude", "intensity"),   # one panel each
        "normalise": True,                          # each frequency divided by its own max
    },

    "export": {
        "save_csv": True,
        "export_directory": None,     # None = '<data_directory>/knife_edge_analysis'
        "save_figure": True,
    },
}


def run_knife_edge(configuration):
    """Load, tabulate, fit. Returns ``(table, edge_fits)``; ``edge_fits`` is [] if disabled."""
    files, skipped = knife_edge.load_knife_edge_files(
        configuration["data_directory"],
        exclude_filename_substrings=configuration["exclude_filename_substrings"])
    for filename, reason in skipped:
        print(f"[knife_edge] skipped {filename}: {reason}")
    table = knife_edge.build_knife_edge_table(files, configuration["frequencies_thz"],
                                              **configuration["transform"])
    knife_edge.print_table(table)

    edge_fits = []
    if configuration["fit"]["enabled"]:
        edge_fits = knife_edge.fit_all_frequencies(
            table,
            minimum_covered_fraction=configuration["fit"]["minimum_covered_fraction"],
            maximum_reduced_chi_square=configuration["fit"]["maximum_reduced_chi_square"])
        knife_edge.print_fits(edge_fits)
    return table, edge_fits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", help="overrides config['data_directory']")
    parser.add_argument("--no-graph", action="store_true", help="do not open the figure window")
    arguments = parser.parse_args(argv)

    configuration = dict(config)
    if arguments.directory:
        configuration["data_directory"] = arguments.directory
    if not os.path.isdir(configuration["data_directory"]):
        print(f"[knife_edge] not a directory: {configuration['data_directory']}", file=sys.stderr)
        return 2

    try:
        table, edge_fits = run_knife_edge(configuration)
    except ValueError as error:
        print(f"[knife_edge] {error}", file=sys.stderr)
        return 2

    export_config = configuration["export"]
    export_directory = export_config["export_directory"] or os.path.join(
        configuration["data_directory"], "knife_edge_analysis")
    if export_config["save_csv"]:
        path = knife_edge.write_table_csv(table, os.path.join(export_directory,
                                                              "knife_edge_table.csv"))
        print(f"[knife_edge] table written to {path}")

    if export_config["save_figure"] or not arguments.no_graph:
        if arguments.no_graph:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = knife_edge.plot_knife_edge(table, edge_fits, **configuration["display"])
        if export_config["save_figure"]:
            os.makedirs(export_directory, exist_ok=True)
            figure_path = os.path.join(export_directory, "knife_edge.png")
            figure.savefig(figure_path, dpi=130)
            print(f"[knife_edge] figure written to {figure_path}")
        if not arguments.no_graph:
            plt.show()
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
