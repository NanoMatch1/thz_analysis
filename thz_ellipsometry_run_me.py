"""THz ellipsometry driver -- self-referenced reflection by switching the emitter magnet.

Isotropic mode (plan sec. 6): rotate the spintronic emitter's magnet through 0/90/180/270 deg,
cycling the states, at one fixed probe setting; leave every THz optic and the GaP crystal where
they are. The ratio of the p and s channels is self-referenced -- everything that multiplies both
(sample height, gain, roughness loss) cancels -- and one instrument constant, from gold or from
probe rotation, turns it into rho = r_p/r_s and then n, k. That is rank 2 of the Jones matrix:
COMPLETE for an isotropic sample, rank-deficient for an anisotropic one, which needs the
generalised (two probe settings) mode still to come.

Run it against real data:

    .venv/bin/python thz_ellipsometry_run_me.py

Each processing step's figures open as the run reaches it (config['general']['inspect']), and
every bundle keeps them; reopen a finished run with ``thz_ellipsometry_view.py <bundle>``.

Or against a synthetic dataset, with no bench and no files, to check the chain end to end:

    .venv/bin/python thz_ellipsometry_run_me.py --simulate hr_silicon
    .venv/bin/python thz_ellipsometry_run_me.py --simulate doped_silicon

Measurement plan (authoritative):     explorations/thz_ellipsometry/
                                      thz_tds_ellipsometry_self_referencing_plan.md
Implementation plan:                  docs/THZ_ELLIPSOMETRY_IMPLEMENTATION_PLAN.md
Design rationale and the error budget: reports/thz_ellipsometry_prospecting_report.md
Silicon validation gates:             docs/ELLIPSOMETRY_MVP_PLAN.md
Findings:                             reports/CNT_measurement_lab_notebook.md, F33-F37
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

from thz_ellipsometry.adapters.export import save_run
from thz_ellipsometry.adapters.inspection import StepwiseInspector
from thz_ellipsometry.adapters.report import format_run_report, plot_run
from thz_ellipsometry.adapters.stages import run_ellipsometry
from thz_ellipsometry.adapters.synthetic_files import write_accumulation_files
from thz_ellipsometry.core import materials
from thz_ellipsometry.core.simulate import interleaved_schedule

DATA_ROOT = os.environ.get("THZ_DATA_ROOT", "/home/match/data")


# ── Configuration ───────────────────────────────────────────────────────────
config: dict = {
    # Registered acquisition mode (thz_ellipsometry.adapters.stages.ACQUISITION_MODES).
    # 'isotropic': magnet states at one fixed probe setting -> rho -> n, k (plan sec. 6).
    "mode": "isotropic",
    "data": {
        # One directory per sample: its magnet-state files plus the gold reference measured in
        # the SAME mount and at the SAME size as the sample (a reference larger than the sample
        # breaks the truncation cancellation, F37).
        "directory": os.path.join(DATA_ROOT, "ellipsometry/2026-10-XX_silicon/hr_silicon"),
    },
    "geometry": {
        # 45 deg: specced by Samuel because the instrument is multi-user and a 140-160 deg
        # two-arm layout does not fit right now. What 45 costs is sensitivity to k (~1.8x
        # worse than at 70) and conditioning on near-mirror samples; what it buys is a much
        # easier geometric budget (shared-tilt tolerance 3.0 deg vs 0.23 at 70).
        "incidence_angle_deg": 45.0,
        # Stated mechanical uncertainty. NOT folded into the noise bars: an angle error is a
        # systematic (eps' = A eps + B, plan sec. 8.1), reported separately as the shift it
        # would cause. Near 45 deg it is ~5% in n per degree.
        "incidence_angle_uncertainty_deg": 0.1,
        # {THz polarisation (deg from p): magnet READING that gives it}, from the wire-grid
        # nulls (bench_run_me null; plan sec. 4.4). One entry is a pure offset, e.g. {0: 86.3};
        # all four states, e.g. {0: 86.3, 90: 176.8, 180: 266.0, 270: 356.9}, also remove
        # per-state errors of the scale. These cannot be fitted from the sample data (one
        # complex redundancy per frequency), and an error left in MIXES the channels (a Moebius
        # map of rho) rather than scaling them, so the gold calibration does not absorb it.
        "magnet_calibration": {0.0: 244.54, 90.0: 337.12, 180.0: 64.77, 270.0: 156.24},
        "index_incident": 1.0,
    },
    "acquisition": {
        "angle_token": "mag",           # filename grammar: doped-si_mag=090_cyc=03.acc
        "probe_token": "probe",         # optional; files without it use detection.probe_azimuth_deg
        "filename_delimiter": "_",
        # role -> filename words; a file matching none is the sample. The angle reference is
        # opt-in by an explicit word because silicon is ALSO a legitimate sample.
        "roles": {
            "angle_reference": ["angleref", "angle-ref", "angle_ref"],
            "channel_reference": ["gold", "mirror"],
        },
        # Fit a polarisation-independent background: equivalent to the plan's +/-M
        # differencing for 0/90/180/270, and valid for unpaired angle sets too.
        "background_term": True,
        # Rows of the harmonic fit: 'acquisition' = one per file (its averaged trace), 'scan' =
        # one per repeat scan. With 'scan' the drift INSIDE each file -- where the polarisation
        # is fixed, so any change is drift -- is measured directly, and 'auto' below becomes a
        # purge-settling trend over the block + a short transient after each opening of the
        # box ('segment_settling') with a gain + tilt ramp. First real block (2026-10-07):
        # reduced chi-square 11.8/3.5 -> 1.3/1.2. Its one assumption is that an opening leaves
        # no lasting timing STEP; only a palindrome checks that.
        "rows": "scan",
        # Purge drift between magnet states does not cancel, so it is modelled over the REAL
        # elapsed time (F35, F40): a delay (gas exchange) plus a gain and a spectral TILT (water;
        # unresolved lines on a short record look like a log-amplitude tilt, linear in f).
        # 'auto' picks by block length: 4 acquisitions -> linear delay + 'tilt_ramp';
        # 8+ (the palindrome 0,90,180,270,270,180,90,0) -> exponential 'settling' delay and
        # 'tilt_settling' gain + tilt (rates fitted): removes the measured purge transient 10 min
        # after closing the box.
        # USE THE PALINDROME: a single pass can only separate drift from the channel ratio when
        # C ~ +1 (F40); the drift_separable diagnostic reports when it could not.
        # Explicit choices: drift 'none'|'linear_ramp'|'settling'|'per_acquisition'|
        # 'segment_settling' (rows='scan' only);
        # amplitude 'none'|'linear_ramp'|'tilt_ramp'|'tilt_settling'|'per_acquisition'.
        "drift_model": "auto",
        "amplitude_model": "auto",
    },
    "detection": {
        # Probe polarisation from the GaP [001] axis. 31.72 deg balances the two channels;
        # 0 and 45 deg are DEGENERATE (one channel blind). This is the fixed isotropic-mode
        # setting, and the primary setting when the sample is also measured at others.
        "probe_azimuth_deg": 31.72,
        # Crystal mounting: [001] measured from p. The plan mounts [001] along s (90 deg).
        # Only the probe_rotation calibration uses it (as the starting guess for the fit).
        "crystal_001_from_p_deg": 90.0,
        "fit_probe_offset": False,      # needs >= 3 probe settings
    },
    "calibration": {
        # Registered source (thz_ellipsometry.core.calibration_sources):
        #   'gold_reference'  gold swapped into the focus (plan sec. 5.1)
        #   'probe_rotation'  the sample itself at two probe settings; nothing in the THz path
        #                     moves (commit c4817b6). Needs probe=... files.
        #   'stored'          replay stored_channel_ratio
        "channel": "gold_reference",
        "reference_material": "gold",
        "stored_channel_ratio": None,
        # 'mechanical' (the angle above is used) or 'fit_from_reference' (fitted on the
        # angle-reference files, plan sec. 5.2). For the silicon VALIDATION keep it mechanical:
        # fitting it on HR-Si consumes HR-Si as a calibrator, so it can no longer validate.
        "incidence_angle": "mechanical",
        "angle_reference_material": "hr_silicon",
    },
    "noise": {
        # Repeat-scan noise model per acquisition -> fit weights and error bars on n, k.
        "enabled": True,
        "minimum_scans": 3,
    },
    "preprocess": {
        # Flat-top Tukey, not Hann: under a sloped window a drifting pulse also changes
        # amplitude, which breaks the drift model for any pulse off the exact window centre
        # (Hann + a strong background gave reduced chi-square 9 at 30 fs drift; Tukey 1.1).
        "window_shape": "tukey",
        "taper_fraction": 0.5,          # central half flat
        "window_half_width_ps": 3.0,
        # Where the window runs past the record (records start ~2.4 ps before the pulse), it is
        # truncated, not resized, and the record's own ends are ramped to zero over this.
        "edge_taper_ps": 0.5,
        "baseline_fraction": 0.1,
        "pad_factor": 4,
        "window_centre_ps": None,       # None = common centre from the mean trace
    },
    "band": {
        # Three independent arguments converge on 1-3 THz: Drude information content,
        # diffraction blur set by sample size, and near-mirror conditioning.
        "frequency_min_thz": 0.8,
        "frequency_max_thz": 3.0,
        "minimum_relative_amplitude": 0.02,
        # Both channels must clear this against the measured noise (plan sec. 6.2).
        "minimum_signal_to_noise": 10.0,
    },
    "inversion": {
        # Out-of-plane tilt does NOT cancel against gold (it rotates the p/s frame). For a
        # DISPERSIVE sample it is fittable from the sample's own data (commit 582b43e); for a
        # flat one (HR-Si) it is degenerate with the channel ratio, so leave this off there.
        "fit_out_of_plane_tilt": False,
        "tilt_model": "drude",          # registered in thz_ellipsometry.core.tilt
        "tilt_fixed_parameters": {},    # e.g. {"eps_inf": 11.7} for doped silicon
        "blur": {
            # Needs a knife-edge number. A 50%-wrong correction is worse than none.
            "enabled": False,
            "angular_spread_deg": None,
        },
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
        # Show each processing step's figures as the run reaches it (needs show_graph), and wait
        # for them to be closed before going on: "all", None, or a list of checkpoints from
        # raw_traces, windowing, spectra, harmonic_fit, calibration, result. Every run saves all
        # of them in its bundle regardless; view a bundle again with thz_ellipsometry_view.py.
        # How to read them: docs/ELLIPSOMETRY_HARMONIC_FIT_TUTORIAL.md
        "inspect": "all",
    },
    "save": {
        # At the END of the run, write a .thzbundle next to the data (recipe + report + results
        # + arrays + figure, with code commits and input-file hashes) and index it in the
        # catalogue. Notes are asked for THEN, with what the run knows pre-filled.
        "enabled": True,
        "directory": None,              # None = the data directory
        # 'auto' prompts only on an interactive terminal; 'config' uses "notes" below; 'none'.
        # THZ_NOTES_MODE in the environment overrides (e.g. none on a batch machine).
        "notes_mode": "auto",
        "notes": "",                    # fallback / non-interactive notes
    },
}


# ── Simulation mode ─────────────────────────────────────────────────────────
SIMULATED_SAMPLES = {
    "hr_silicon": lambda frequencies: materials.high_resistivity_silicon_index(frequencies),
    "doped_silicon": lambda frequencies: materials.doped_silicon_index(frequencies, 1.0),
}

#: Magnet states of the plan's acquisition (sec. 4.2), cycled rather than blocked.
SIMULATED_MAGNET_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)


def build_simulated_dataset(directory, sample_name, *, incidence_angle_deg,
                            emitter_angles_deg=SIMULATED_MAGNET_ANGLES_DEG, cycles=3,
                            scans_per_acquisition=4, seconds_per_scan=10.0,
                            relative_noise=0.002, drift_span_fs=30.0, amplitude_drift=0.0,
                            background_relative=0.05, out_of_plane_tilt_deg=0.0,
                            magnet_calibration=None, probe_azimuth_deg=None,
                            second_probe_azimuth_deg=None, crystal_001_from_p_deg=90.0,
                            angle_reference=False, drift_within_acquisition=False):
    """Write a synthetic sample + gold reference measurement, in the real .acc format.

    This is how the whole driver gets exercised without beam time: the files go through the
    real loader, the real noise model, the real preprocessing and the real analysis. The sample
    and the gold are each an interleaved schedule of magnet states with repeat scans, a drift
    over their own elapsed time, and a non-magnetic background. With
    ``second_probe_azimuth_deg`` the sample is also written at a second probe setting (for the
    probe-rotation calibration); with ``angle_reference`` an HR-Si angle reference is added.
    """
    probe_deg = (config["detection"]["probe_azimuth_deg"] if probe_azimuth_deg is None
                 else probe_azimuth_deg)
    if magnet_calibration is None:
        # Write the readings the configured table expects, so the analysis -- which applies that
        # table -- recovers the planted polarisations.
        magnet_calibration = config["geometry"].get("magnet_calibration")
    schedule = interleaved_schedule(
        emitter_angles_deg, cycles=cycles,
        seconds_per_acquisition=scans_per_acquisition * seconds_per_scan)
    block_seconds = float(schedule.elapsed_seconds[-1]) + scans_per_acquisition * seconds_per_scan
    shared = dict(schedule=schedule, scans_per_angle=scans_per_acquisition,
                  seconds_per_scan=seconds_per_scan, angle_token="mag",
                  magnet_calibration=magnet_calibration,
                  incidence_angle_rad=np.deg2rad(incidence_angle_deg),
                  crystal_orientation_rad=np.deg2rad(crystal_001_from_p_deg),
                  relative_noise=relative_noise, drift_span_s=drift_span_fs * 1e-15,
                  amplitude_drift=amplitude_drift,
                  background_relative=background_relative,
                  drift_within_acquisition=drift_within_acquisition)
    write_accumulation_files(directory, index_sample_function=SIMULATED_SAMPLES[sample_name],
                             sample_name=sample_name, probe_azimuth_deg=probe_deg,
                             out_of_plane_tilt_rad=np.deg2rad(out_of_plane_tilt_deg), seed=0,
                             **shared)
    write_accumulation_files(directory, index_sample_function=materials.gold_index,
                             sample_name="ref-gold", probe_azimuth_deg=probe_deg, seed=1,
                             time_offset_seconds=block_seconds, **shared)
    if second_probe_azimuth_deg is not None:
        write_accumulation_files(directory, index_sample_function=SIMULATED_SAMPLES[sample_name],
                                 sample_name=sample_name, probe_token="probe",
                                 probe_azimuth_deg=second_probe_azimuth_deg,
                                 out_of_plane_tilt_rad=np.deg2rad(out_of_plane_tilt_deg),
                                 seed=2, time_offset_seconds=2 * block_seconds, **shared)
    if angle_reference:
        write_accumulation_files(directory,
                                 index_sample_function=SIMULATED_SAMPLES["hr_silicon"],
                                 sample_name="angleref-hr-silicon", probe_azimuth_deg=probe_deg,
                                 seed=3, time_offset_seconds=3 * block_seconds, **shared)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulate", choices=sorted(SIMULATED_SAMPLES),
                        help="synthesise this sample instead of reading real data")
    parser.add_argument("--directory", help="override config['data']['directory']")
    parser.add_argument("--no-graph", action="store_true", help="headless; skip the figure")
    parser.add_argument("--no-save", action="store_true", help="do not write a result bundle")
    parser.add_argument("--notes", help="notes to save with the bundle (skips the prompt)")
    arguments = parser.parse_args(argv)

    if arguments.no_graph:
        config["general"]["show_graph"] = False

    temporary = None
    if arguments.simulate:
        temporary = tempfile.mkdtemp(prefix="ellipsometry_simulated_")
        build_simulated_dataset(
            temporary, arguments.simulate,
            incidence_angle_deg=config["geometry"]["incidence_angle_deg"],
            crystal_001_from_p_deg=config["detection"]["crystal_001_from_p_deg"],
            # Drift runs through the scans as on the bench; a staircase (constant per file)
            # is exactly the lasting step at each opening that per-scan rows cannot see.
            drift_within_acquisition=True)
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

    general = config["general"]
    inspector = (StepwiseInspector(general.get("inspect"))
                 if general.get("show_graph") and general.get("inspect") else None)
    outcome = run_ellipsometry(config, observer=inspector)
    print(format_run_report(outcome))

    if config["general"]["show_graph"] or config["general"]["save_figure"]:
        plot_run(outcome, show=config["general"]["show_graph"],
                 save_path=config["general"]["save_figure"])

    save = config.get("save") or {}
    if save.get("enabled", True) and not arguments.no_save:
        save_run(outcome, directory=save.get("directory"),
                 notes_mode="config" if arguments.notes else save.get("notes_mode", "auto"),
                 config_notes=arguments.notes or save.get("notes", ""),
                 producer=os.path.basename(__file__))

    if outcome.report is not None and not outcome.report.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
