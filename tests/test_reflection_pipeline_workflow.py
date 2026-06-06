"""Headless full-workflow test for the window-geometry reflection pipeline.

Builds synthetic .acc files (a SiO2-only reference and a sample with a known
lossless index pressed on the window), runs the whole adapter chain, and asserts
the pipeline recovers the known sample index. Echoes are placed at *identical*
timing so there is no phase ramp — this isolates the geometry/reference/inversion
plumbing from the (separately handled) timing-correction step.

Dependency-free (no pytest); run with the project venv:
    .venv/Scripts/python.exe tests/test_reflection_pipeline_workflow.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

import thz_core as core
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


N_SIO2 = 1.95
THETA_EXT_DEG = 45.0
N_SAMPLE_TRUTH = 3.0  # lossless -> real reflection, clean recovery
FRONT_PS = 153.4
ECHO_PS = 164.3


def _gaussian(t_ps, centre, width=0.25):
    return np.exp(-((t_ps - centre) ** 2) / (2 * width**2))


def _write_acc(path, name, echo_amplitude):
    """Write a single-scan .acc with a front pulse + a scaled echo (same timing)."""
    t_ps = np.arange(151.0, 169.0, 0.05)
    front = 1.0 * _gaussian(t_ps, FRONT_PS)
    echo = echo_amplitude * _gaussian(t_ps, ECHO_PS)
    y = front + echo
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t:.6f} {e:.10e}" for t, e in zip(t_ps, y)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_dataset(tmpdir):
    theta_int = float(np.real(core.snell_refracted_angle(np.deg2rad(THETA_EXT_DEG), 1.0, N_SIO2)))
    r_air = core.fresnel_reflection_s(N_SIO2, 1.0, theta_int)            # ~ +0.44
    r_sample = core.fresnel_reflection_s(N_SIO2, N_SAMPLE_TRUTH, theta_int)  # negative (n>n_sio2)

    # Reference echo carries r_{SiO2->air}; sample echo carries r_{SiO2->sample}.
    # H = sample/ref, then r_recovered = r_air * H = r_{SiO2->sample}.
    _write_acc(os.path.join(tmpdir, "reference_sio2_set_0.acc"), "reference_sio2_set_0", float(r_air.real))
    _write_acc(os.path.join(tmpdir, "sample_film-1_set_0.acc"), "sample_film-1_set_0", float(r_sample.real))
    return r_air, r_sample, theta_int


def test_window_reflection_pipeline_recovers_known_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        r_air, r_sample, theta_int = _build_dataset(tmpdir)

        dataset = DataSet(tmpdir)
        dataset.load_all_data(case_insensitive=True)
        dataset.group_files(keywords=["type"])

        # Pairing: the sample must resolve the SiO2 reference.
        samp_name = "sample_film-1_set_0.acc"
        ref = dataset.get_reference(samp_name, ref_type="reference")
        assert ref is not None, "sample failed to pair with the SiO2 reference"

        gates = {"first_reflection": (151.2, 158.5), "echo": (159.5, 168.8)}
        thz.segment_reflections(dataset, segments=gates, active="echo")
        thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
        thz.fft_spectrum(dataset)
        thz.transfer_function(dataset, ref_type="reference")
        thz.invert_nk_reflection(
            dataset, geometry="window", theta_deg=THETA_EXT_DEG, n_window=N_SIO2,
        )

        samp = dataset.data[samp_name]
        pd_ = samp.processing_dict

        # Internal angle correctly computed by Snell.
        assert abs(np.rad2deg(pd_["theta_internal_rad"]) - 21.26) < 0.2

        freq = pd_["fft_freq"]
        n = pd_["n"]
        k = pd_["k"]
        f_thz = freq * 1e-12
        band = (f_thz > 0.5) & (f_thz < 3.0) & np.isfinite(n)
        assert band.sum() > 10, "too few finite bins to assess recovery"

        n_mean = float(np.nanmean(n[band]))
        k_mean = float(np.nanmean(k[band]))
        # Lossless sample with no timing offset -> clean recovery of n=3, k~0.
        assert abs(n_mean - N_SAMPLE_TRUTH) < 0.05, f"n recovered {n_mean:.3f}, expected 3.0"
        assert k_mean < 0.05, f"k recovered {k_mean:.3f}, expected ~0"


def test_recovered_reflection_has_pi_phase_for_high_index_sample():
    """A sample of higher index than SiO2 must give a sign-flipped (negative) r."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _build_dataset(tmpdir)
        dataset = DataSet(tmpdir)
        dataset.load_all_data(case_insensitive=True)
        dataset.group_files(keywords=["type"])
        gates = {"first_reflection": (151.2, 158.5), "echo": (159.5, 168.8)}
        thz.segment_reflections(dataset, segments=gates, active="echo")
        thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
        thz.fft_spectrum(dataset)
        thz.transfer_function(dataset, ref_type="reference")
        thz.invert_nk_reflection(
            dataset, geometry="window", theta_deg=THETA_EXT_DEG, n_window=N_SIO2,
        )
        samp = dataset.data["sample_film-1_set_0.acc"]
        pd_ = samp.processing_dict
        r = pd_["reflection_r"]
        mask = pd_["transfer_mask"]
        r_band = r[mask & np.isfinite(r)]
        assert np.mean(r_band.real) < 0, "high-index sample should give negative (pi-flipped) r"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {error}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
