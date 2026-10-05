"""THz ellipsometry bench tools -- answers to setup questions, run on a directory as data arrive.

    .venv/bin/python thz_ellipsometry_bench_run_me.py null  <dir>   # wire-grid magnet zero
    .venv/bin/python thz_ellipsometry_bench_run_me.py live  <dir>   # has the purge settled?
    .venv/bin/python thz_ellipsometry_bench_run_me.py live  <dir> --watch 60
    .venv/bin/python thz_ellipsometry_bench_run_me.py hwp   <dir>   # half-wave-plate beam walk

The subcommands are generated from the tool registry in thz_ellipsometry.adapters.bench_tools;
the order of work on the bench is in reports/ellipsometry_bench_plan_2026-10-06.md.
"""

from __future__ import annotations

import argparse
import sys
import time

from thz_ellipsometry.adapters.bench_tools import BENCH_TOOLS, run_bench_tool

# ── Configuration ───────────────────────────────────────────────────────────
config: dict = {
    # Same preprocessing as the analysis driver, so a bench number means the same thing there.
    "preprocess": {
        "window_shape": "tukey",
        "taper_fraction": 0.5,
        "window_half_width_ps": 3.0,
        "baseline_fraction": 0.1,
        "pad_factor": 4,
    },
    "band": {"frequency_min_thz": 0.8, "frequency_max_thz": 3.0},
    "null": {
        # wgp_grid=s_null=0_mag=086.acc: one sweep per null, labelled with the polarisation it
        # sets (grid passing s -> null=0 and null=180; passing p -> null=90 and null=270).
        "angle_token": "mag",
        "group_token": "null",
        "filename_contains": None,
    },
    "live": {
        "angle_token": "mag",           # scans are grouped by magnet state
        "filename_contains": None,
        "window_scans": 3,              # the rate is fitted over the latest scans
        # Settled below this; tune it on day 1. What it costs: drift accumulated BETWEEN magnet
        # states inside a block is differential, and 1.5 fs of it is ~0.9% in rho at 1 THz
        # (F35). 1 fs/min over a 10 min step is 10 fs; the drift-ramp fit removes the linear
        # part of that, so the limit guards against what is not linear (the fast transient).
        "rate_limit_fs_per_minute": 1.0,
    },
    "hwp": {
        "plate_token": "hwp",           # hwp-gold_hwp=090_mag=045.acc
        "filename_contains": None,
        "reference_plate_deg": None,    # None = the lowest plate angle measured
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="tool", required=True)
    for name, entry in sorted(BENCH_TOOLS.items()):
        command = subcommands.add_parser(name, help=entry.summary)
        command.add_argument("directory")
        command.add_argument("--watch", type=float, metavar="SECONDS",
                             help="re-run every SECONDS until interrupted")
    arguments = parser.parse_args(argv)

    while True:
        try:
            _, report = run_bench_tool(arguments.tool, arguments.directory, config)
            print(report, flush=True)
        except (ValueError, FileNotFoundError, NotADirectoryError) as error:
            print(f"[{arguments.tool}] {error}", file=sys.stderr, flush=True)
            if not arguments.watch:
                return 2
        if not arguments.watch:
            return 0
        time.sleep(arguments.watch)


if __name__ == "__main__":
    raise SystemExit(main())
