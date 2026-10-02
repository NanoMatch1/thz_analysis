"""THz ellipsometry -- Phase 1 driver (isotropic samples, emitter polarisation only).

Rotate the THz polarisation at the emitter; leave the GaP crystal where it is. That is rank 2
of the Jones matrix, which is COMPLETE for an isotropic sample such as silicon and
rank-deficient only for an anisotropic one. Anisotropy needs a second detection azimuth and is
Phase 2.

Run it against real data:

    .venv/bin/python thz_ellipsometry_run_me.py

Or against a synthetic dataset, with no bench and no files, to check the chain end to end:

    .venv/bin/python thz_ellipsometry_run_me.py --simulate hr_silicon
    .venv/bin/python thz_ellipsometry_run_me.py --simulate doped_silicon

Design rationale and the error budget: reports/thz_ellipsometry_prospecting_report.md
Build spec and validation gates:      docs/ELLIPSOMETRY_MVP_PLAN.md
Findings:                             reports/CNT_measurement_lab_notebook.md, F33-F37
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

from thz_ellipsometry.adapters.report import format_run_report, plot_run
from thz_ellipsometry.adapters.stages import run_ellipsometry
from thz_ellipsometry.adapters.synthetic_files import write_accumulation_files
from thz_ellipsometry.core import materials

DATA_ROOT = os.environ.get("THZ_DATA_ROOT", "/home/match/data")


# ── Configuration ───────────────────────────────────────────────────────────
config: dict = {
    "data": {
        # One directory per sample, holding its polarisation series plus the gold reference
        # measured in the SAME mount and at the SAME size as the sample (see the report, 4.7:
        # a reference larger than the sample breaks the truncation cancellation).
        "directory": os.path.join(DATA_ROOT, "ellipsometry/2026-10-XX_silicon/hr_silicon"),
    },
    "geometry": {
        # 45 deg: specced by Samuel because the instrument is multi-user and a 140-160 deg
        # two-arm layout does not fit right now. Validated -- the pipeline recovers silicon at
        # 45 deg. What 45 costs is sensitivity to k (the floor is ~1.8x worse than at 70) and
        # conditioning on near-mirror samples, neither of which this validation needs.
        # What 45 BUYS is a much easier geometric budget: the shared-tilt tolerance is 3.0 deg
        # and the emitter-offset tolerance 0.78 deg, versus 0.23 deg each at 70.
        # Set mechanically, with a visible laser off the polished wafer. Silicon reflects
        # visible light, so the no-alignment-handle problem of the CNT work does not apply.
        "incidence_angle_deg": 45.0,
        # The emitter's angular zero relative to the PLANE OF INCIDENCE. Not absorbed by the
        # gold calibration -- it mixes the two channels rather than scaling them. Leaving it
        # uncorrected costs about 0.009 in |N| per 0.1 deg at 70 deg incidence, so keep it
        # inside ~0.2 deg or measure it once during setup with a wire grid.
        "emitter_offset_deg": 0.0,
        "index_incident": 1.0,
    },
    "polarization": {
        "angle_token": "pol",          # filename grammar: hr-silicon_pol=15.acc
        "filename_delimiter": "_",
        # The GaP [001] axis versus the probe polarisation. 31.72 deg balances the two
        # channels. 0 and 45 deg are DEGENERATE -- one channel is blind and rho cannot be
        # measured at all, which is where the crystal currently sits.
        "probe_azimuth_deg": 31.72,
        # 'linear_ramp' matches our measured error, a slow ordered drift (F35), and needs at
        # least THREE emitter angles to be identifiable. Four (0/45/90/135) is the sweet spot:
        # drift-immune, one spare degree of freedom for the residual quality flag, and no worse
        # than two angles at equal total measurement time. Interleave the acquisition order so
        # the drift is common-mode between neighbouring settings.
        "drift_model": "linear_ramp",
    },
    "reference": {
        "vocabulary": ["gold", "mirror"],
        "index": "gold",
        "channel_ratio": None,          # set only to replay a stored calibration
    },
    "preprocess": {
        "window_half_width_ps": 3.0,
        "baseline_fraction": 0.1,
        "pad_factor": 4,
        "window_centre_ps": None,       # None = common centre from the mean trace
    },
    "band": {
        # Three independent arguments converge on 1-3 THz: Drude information content,
        # diffraction blur set by sample size, and near-mirror conditioning.
        "frequency_min_thz": 0.8,
        "frequency_max_thz": 3.0,
        # Drop bins whose amplitude has fallen below this fraction of the series
        # peak; dead bins otherwise reach the calibration as pure noise.
        "minimum_relative_amplitude": 0.02,
    },
    "blur": {
        # Needs a knife-edge number. A 50%-wrong correction is worse than none, so this stays
        # off until the angular spread has been measured (OQ10/OQ12).
        "enabled": False,
        "angular_spread_deg": None,
    },
    "validation": {
        "expect": "hr_silicon",         # or None to skip; 'gold' also available
        "cross_check_material": "hr_silicon",   # fit theta as a CROSS-CHECK, never applied
        "branch_reference_index": materials.SILICON_HIGH_RESISTIVITY_INDEX,
        "tolerance_n": 0.02,
        "tolerance_k": 0.05,
        "tolerance_angle_deg": 0.1,
    },
    "general": {
        "show_graph": True,
        "save_figure": None,
    },
}


# ── Simulation mode ─────────────────────────────────────────────────────────
SIMULATED_SAMPLES = {
    "hr_silicon": lambda frequencies: materials.high_resistivity_silicon_index(frequencies),
    "doped_silicon": lambda frequencies: materials.doped_silicon_index(frequencies, 1.0),
}


def build_simulated_dataset(directory, sample_name, *, incidence_angle_deg,
                            emitter_angles_deg, relative_noise=0.002, drift_span_fs=30.0):
    """Write a synthetic sample + gold reference series, in the real .acc format.

    This is how the whole driver gets exercised without beam time: the files go through the
    real loader, the real preprocessing and the real analysis.
    """
    angles_rad = np.deg2rad(emitter_angles_deg)
    shared = dict(emitter_angles_rad=angles_rad,
                  incidence_angle_rad=np.deg2rad(incidence_angle_deg),
                  relative_noise=relative_noise, drift_span_s=drift_span_fs * 1e-15)
    write_accumulation_files(directory, index_sample_function=SIMULATED_SAMPLES[sample_name],
                             sample_name=sample_name, **shared)
    write_accumulation_files(directory, index_sample_function=materials.gold_index,
                             sample_name="ref-gold", seed=1, **shared)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulate", choices=sorted(SIMULATED_SAMPLES),
                        help="synthesise this sample instead of reading real data")
    parser.add_argument("--directory", help="override config['data']['directory']")
    parser.add_argument("--no-graph", action="store_true", help="headless; skip the figure")
    arguments = parser.parse_args(argv)

    if arguments.no_graph:
        config["general"]["show_graph"] = False

    temporary = None
    if arguments.simulate:
        temporary = tempfile.mkdtemp(prefix="ellipsometry_simulated_")
        build_simulated_dataset(
            temporary, arguments.simulate,
            incidence_angle_deg=config["geometry"]["incidence_angle_deg"],
            emitter_angles_deg=np.linspace(0.0, 180.0, 4, endpoint=False))
        config["data"]["directory"] = temporary
        config["validation"]["expect"] = (
            arguments.simulate if arguments.simulate in ("hr_silicon",) else None)
        print(f"[simulate] wrote a synthetic {arguments.simulate} dataset to {temporary}")
    elif arguments.directory:
        config["data"]["directory"] = arguments.directory

    if not os.path.isdir(config["data"]["directory"]):
        print(f"[thz_ellipsometry_run_me] data directory does not exist: "
              f"{config['data']['directory']}\n"
              f"   Point config['data']['directory'] at a real directory, set THZ_DATA_ROOT, "
              f"or run with --simulate hr_silicon.", file=sys.stderr)
        return 2

    outcome = run_ellipsometry(config)
    print(format_run_report(outcome))

    if config["general"]["show_graph"] or config["general"]["save_figure"]:
        plot_run(outcome, show=config["general"]["show_graph"],
                 save_path=config["general"]["save_figure"])

    if outcome.report is not None and not outcome.report.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
