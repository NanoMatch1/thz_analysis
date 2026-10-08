"""View a finished ellipsometry run step by step, from its bundle -- nothing is rerun.

    .venv/bin/python thz_ellipsometry_view.py <bundle>                    # every step, in order
    .venv/bin/python thz_ellipsometry_view.py <bundle> --step windowing   # one step (repeatable)
    .venv/bin/python thz_ellipsometry_view.py <bundle> --list             # what can be shown
    .venv/bin/python thz_ellipsometry_view.py <bundle> --save <dir>       # write PNGs, no window

``<bundle>`` is a ``.thzbundle`` directory written by ``thz_ellipsometry_run_me.py`` (or the
``inspection.npz`` inside it). Each step's figures open together; close them to go on to the
next. The figures are the ones the run showed (``config['general']['inspect']``), redrawn from
the arrays the bundle stored. How to read them: docs/ELLIPSOMETRY_HARMONIC_FIT_TUTORIAL.md.
"""

from __future__ import annotations

import argparse
import os
import sys

from thz_ellipsometry.adapters import inspection
from thz_ellipsometry.adapters.stages import CHECKPOINTS


def inspection_path(bundle):
    """The inspection.npz of a bundle directory (or the file itself)."""
    path = bundle if bundle.endswith(".npz") else os.path.join(bundle,
                                                               inspection.INSPECTION_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {inspection.INSPECTION_FILENAME} at {path}; bundles saved "
                                "before step inspection existed have none -- rerun the analysis")
    return path


def describe_steps():
    lines = []
    for checkpoint in CHECKPOINTS:
        for entry in inspection.figures_at(checkpoint):
            lines.append(f"   {checkpoint:<13} {entry.name:<30} {entry.summary}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", help="a .thzbundle directory, or its inspection.npz")
    parser.add_argument("--step", action="append", choices=CHECKPOINTS,
                        help="show only this step (repeat for several); default all")
    parser.add_argument("--list", action="store_true", help="list the steps and their figures")
    parser.add_argument("--save", metavar="DIRECTORY",
                        help="write the figures as PNGs there instead of showing them")
    arguments = parser.parse_args(argv)

    if arguments.list:
        print(describe_steps())
        return 0
    try:
        record = inspection.load_inspection(inspection_path(arguments.bundle))
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"[thz_ellipsometry_view] {error}", file=sys.stderr)
        return 2

    import matplotlib
    if arguments.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = arguments.step or list(CHECKPOINTS)
    for checkpoint in CHECKPOINTS:
        if checkpoint not in steps:
            continue
        figures = inspection.make_inspection_figures(record, (checkpoint,))
        if arguments.save:
            for path in inspection.save_figures(figures, arguments.save):
                print(f"[thz_ellipsometry_view] wrote {path}")
        else:
            print(f"[thz_ellipsometry_view] {checkpoint}: {len(figures)} figure(s) -- close "
                  "them to continue", flush=True)
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
