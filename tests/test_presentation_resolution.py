"""Tests for instrument-resolution limiting and frequency-domain uncertainty.

Covers the presentation-quality additions:
  - compute_instrument_resolution: df_res = 1/T_res from the shorter reflection window,
    and the decimation factor k = round(df_res / df_fft_bin_spacing).
  - apply_instrument_resolution: decimates every frequency-indexed array by k when the
    config flag is on, and is a no-op when off.
  - compute_transfer_uncertainty: the ratio error-propagation formula from per-bin SNR.
  - _resolution_marker_stride and the _plot_with_snr_mask helper (headless).

Dependency-free; run with the project venv:
    .venv/Scripts/python.exe tests/test_presentation_resolution.py
"""

import os
import sys
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acquisition_editor import save_acc
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


DT_PS = 0.05
FIRST_PS = 155.0
SECOND_PS = 180.0
FIRST_SEARCH = (146.0, 160.0)
SECOND_SEARCH = (170.0, 188.0)


def _pulse(t_ps, centre, amp, width=0.3):
    x = (t_ps - centre) / width
    return amp * -x * np.exp(-(x**2) / 2.0)


def _write_acc(path, name, n_scans, *, second_amp, rng):
    # Zero-fill between/around the two pulses, mirroring the real acquisition, so the
    # found regions are the two ~compact pulse windows.
    t_ps = np.arange(150.0, 185.0, DT_PS)
    columns = [t_ps]
    scan_headers = []
    for scan_index in range(n_scans):
        clean = _pulse(t_ps, FIRST_PS, 1.0) + _pulse(t_ps, SECOND_PS, second_amp)
        noisy = clean + rng.normal(0.0, 0.01, size=t_ps.size)
        in_first = (t_ps >= FIRST_SEARCH[0]) & (t_ps <= FIRST_SEARCH[1])
        in_second = (t_ps >= SECOND_SEARCH[0]) & (t_ps <= SECOND_SEARCH[1])
        noisy[~(in_first | in_second)] = 0.0
        columns.append(noisy)
        scan_headers.append([f"title {name} acc {scan_index + 1}", "type 0"])
    save_acc({"data": np.column_stack(columns), "scan_headers": scan_headers,
              "header": scan_headers[0]}, path)


def _build_through_transfer(tmpdir, *, n_fft=2000):
    rng = np.random.default_rng(0)
    _write_acc(os.path.join(tmpdir, "reference_x.acc"), "reference_x", 4,
               second_amp=-0.4, rng=rng)
    _write_acc(os.path.join(tmpdir, "sample_x.acc"), "sample_x", 6,
               second_amp=-0.25, rng=rng)
    config = {
        "regions": {"first_reflection": FIRST_SEARCH, "second_reflection": SECOND_SEARCH},
        "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 3},
        "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": n_fft},
        "transfer": {"self_reference": True, "apply_snr_mask": True,
                     "min_ref_amp_rel": 1e-3, "regularization_eps": 1e-30,
                     "unwrap_phase": True},
        "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    }
    dataset = DataSet(tmpdir, config=config)
    dataset.load_all_data(case_insensitive=True)
    thz.build_full_trace_reflection(dataset, config)
    thz.define_reflection_regions(dataset, config)
    thz.subtract_baseline(dataset)
    dataset.group_files(keywords=["type"])
    thz.window_pulses_fixed_width(dataset, half_width_ps=config["window"]["half_width_ps"],
                                  show_graph=False)
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=n_fft)
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=n_fft)
    thz.transfer_function(dataset, ref_type="reference")
    return dataset, config


def test_resolution_matches_shortest_window():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _build_through_transfer(tmpdir)
        block = thz.compute_instrument_resolution(dataset, config)

        # Shortest found region sets T_res; regions were tightened by the builder.
        regions = config["regions"]
        first_len = (regions["first_reflection"][1] - regions["first_reflection"][0]) * 1e-12
        second_len = (regions["second_reflection"][1] - regions["second_reflection"][0]) * 1e-12
        t_res_expected = min(first_len, second_len)
        assert abs(block["t_resolution_s"] - t_res_expected) < 1e-15, block

        assert abs(block["df_resolution_hz"] - 1.0 / t_res_expected) < 1.0, block
        k_expected = max(1, round(block["df_resolution_hz"] / block["df_fft_hz"]))
        assert block["decimation_factor"] == k_expected
        assert block["decimation_factor"] > 1  # n_fft=2000 is heavily oversampled


def test_apply_decimates_all_freq_arrays():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _build_through_transfer(tmpdir)
        thz.compute_instrument_resolution(dataset, config)
        k = config["resolution"]["decimation_factor"]

        sample = dataset.data["sample_x.acc"]
        n_before = len(sample.processing_dict["fft_freq"])
        H_before = sample.processing_dict["transfer_H"].copy()

        config["resolution"]["limit_to_instrument_resolution"] = True
        thz.apply_instrument_resolution(dataset, config)

        n_after = len(sample.processing_dict["fft_freq"])
        assert n_after == len(np.arange(0, n_before, k)), (n_after, n_before, k)
        # Decimation = pick every k-th bin (values untouched, not averaged).
        assert np.allclose(sample.processing_dict["transfer_H"], H_before[::k], equal_nan=True)
        # A frequency-indexed array of a different name came along too.
        assert len(sample.processing_dict["transfer_mask"]) == n_after
        # The 2-D frequency-domain working array was decimated as well.
        assert sample.second_reflection.data.shape[0] == n_after


def _build_single_reflection_through_fft(tmpdir, half_width_ps=3, n_fft=2000):
    """A plain-THzData (single-bounce) dataset windowed + FFT'd — no THzDataReflection."""
    rng = np.random.default_rng(0)
    for name, amp in (("reference_x.acc", 1.0), ("sample_x.acc", 0.6)):
        t_ps = np.arange(150.0, 185.0, DT_PS)
        noisy = _pulse(t_ps, 165.0, amp) + rng.normal(0.0, 0.01, size=t_ps.size)
        save_acc({"data": np.column_stack([t_ps, noisy]),
                  "scan_headers": [[f"title {name} acc 1", "type 0"]],
                  "header": [f"title {name} acc 1", "type 0"]},
                 os.path.join(tmpdir, name))
    config = {
        "window": {"type": "hann", "alpha": 1.0, "half_width_ps": half_width_ps},
        "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": n_fft},
    }
    dataset = DataSet(tmpdir, config=config)
    dataset.load_all_data(case_insensitive=True)
    from dataset_core.data_structures.thz import THzData, THzDataReflection
    assert all(isinstance(o, THzData) and not isinstance(o, THzDataReflection)
               for o in dataset.data.values())
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=half_width_ps,
                                        region_ps=None, show_graph=False)
    thz.fft_spectrum(dataset, n_fft=n_fft)
    return dataset, config


def test_resolution_works_on_single_reflection_thzdata():
    """compute_instrument_resolution must handle plain THzData (was skipping before)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        half_width_ps = 3
        dataset, config = _build_single_reflection_through_fft(tmpdir, half_width_ps=half_width_ps)
        block = thz.compute_instrument_resolution(dataset, config)

        # Did NOT skip: a resolution block was produced.
        assert "t_resolution_s" in block, block
        # T_res = the windowed non-zero support ~ the window length (2*half_width_ps),
        # a hair under because the Hann taper is exactly zero at its endpoints.
        window_length_ps = 2 * half_width_ps
        assert abs(block["t_resolution_s"] * 1e12 - window_length_ps) < 0.3, block
        # df_res is 1/T_res by definition (broadening_factor 1).
        assert abs(block["df_resolution_hz"] - 1.0 / block["t_resolution_s"]) < 1.0, block
        assert block["decimation_factor"] > 1

        # And apply decimates the plain-THzData frequency arrays.
        sample = dataset.data["sample_x.acc"]
        n_before = len(sample.processing_dict["fft_freq"])
        config["resolution"]["limit_to_instrument_resolution"] = True
        thz.apply_instrument_resolution(dataset, config)
        k = block["decimation_factor"]
        assert len(sample.processing_dict["fft_freq"]) == len(np.arange(0, n_before, k))


def test_apply_is_noop_when_flag_off():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _build_through_transfer(tmpdir)
        thz.compute_instrument_resolution(dataset, config)
        sample = dataset.data["sample_x.acc"]
        n_before = len(sample.processing_dict["fft_freq"])
        config["resolution"]["limit_to_instrument_resolution"] = False
        thz.apply_instrument_resolution(dataset, config)
        assert len(sample.processing_dict["fft_freq"]) == n_before


def test_transfer_uncertainty_formula():
    """σ_|H| = |H|·sqrt(10^(-samp_snr/10) + 10^(-ref_snr/10)); σ_phase = the sqrt term."""
    freq_n = 8
    H = (np.linspace(0.5, 2.0, freq_n) + 0.3j).astype(complex)
    samp_snr_db = np.array([30, 25, 20, 15, 10, 6, 3, 0], dtype=float)
    ref_snr_db = np.array([28, 24, 20, 16, 12, 8, 4, 1], dtype=float)

    class _Obj:
        def __init__(self):
            self.processing_dict = {
                "transfer_H": H, "snr_db": samp_snr_db, "ref_snr_db": ref_snr_db,
            }

    class _Dataset:
        def __init__(self):
            self.data = {"sample_x.acc": _Obj()}
            self.config = {}

    dataset = _Dataset()
    thz.compute_transfer_uncertainty(dataset)
    pd = dataset.data["sample_x.acc"].processing_dict

    rel = np.sqrt(10.0 ** (-samp_snr_db / 10.0) + 10.0 ** (-ref_snr_db / 10.0))
    assert np.allclose(pd["transfer_phase_sigma"], rel)
    assert np.allclose(pd["transfer_H_sigma"], np.abs(H) * rel)
    # Error explodes at low SNR (last bin) and is tiny at high SNR (first bin).
    assert pd["transfer_H_sigma"][-1] > pd["transfer_H_sigma"][0] * 20


def test_marker_stride_logic():
    class _DS:
        def __init__(self, cfg):
            self.config = cfg
    # Not computed -> stride 1.
    assert thz._resolution_marker_stride(_DS({})) == 1
    # Oversampled, flag off -> stride = decimation factor.
    assert thz._resolution_marker_stride(
        _DS({"resolution": {"decimation_factor": 15}})) == 15
    # Already decimated -> stride 1.
    assert thz._resolution_marker_stride(
        _DS({"resolution": {"decimation_factor": 15,
                            "limit_to_instrument_resolution": True}})) == 1


def test_plot_helper_runs_headless_with_markers_and_error():
    x = np.linspace(0, 3, 60)
    y = np.sin(x)
    mask = x < 2.0            # upper third untrusted
    yerr = 0.05 * np.ones_like(y)
    fig, ax = plt.subplots()
    line = thz._plot_with_snr_mask(ax, x, y, mask, label="test",
                                   marker_every=5, yerr=yerr)
    # Trusted markers exist and there is at least one shaded untrusted band.
    assert line.get_label() == "test"
    assert len(ax.patches) >= 1  # axvspan patch(es) for the untrusted band
    plt.close(fig)


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
