"""Tests for per-acquisition drift tracking (``dataset_core.adapters.acquisition_tracking``).

Two forms, per the project's testing philosophy:
  * discrete unit tests — header parsing, extraction, each drift metric against a
    KNOWN planted answer, the registry contract, and the summary/export helpers
  * a headless workflow test — the whole script path (load -> extract -> every
    figure) runs on an Agg backend without raising

The metric tests are known-answer tests: a synthetic series is built with a drift
planted in it by construction, and the metric must recover that exact number.
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.data_structures.thz import BaseTHzData, THzData
from dataset_core.adapters import acquisition_tracking as tracking


# ── synthetic builders ───────────────────────────────────────────────────────


def _pulse(time_ps: np.ndarray, centre_ps: float = 2.25, width_ps: float = 0.25) -> np.ndarray:
    """A THz-like pulse: one dominant central extremum with smaller side lobes.

    Second derivative of a Gaussian, deliberately NOT the antisymmetric first
    derivative — that one has two exactly-equal extrema, so ``argmax(|y|)`` picks
    between them on floating-point noise and the window position flips between
    otherwise identical test runs. Real THz pulses have an unambiguous peak, and
    the fixture should too.
    """
    reduced = (time_ps - centre_ps) / width_ps
    return (1.0 - reduced ** 2) * np.exp(-0.5 * reduced ** 2)


def _make_thzdata(
    n_scans: int = 24,
    *,
    amplitude_per_scan=None,
    delay_ps_per_scan=None,
    offset_per_scan=None,
    start_ps: float = 107.5,
    dt_ps: float = 0.05,
    n_time: int = 135,
    noise: float = 0.0,
    title: str = 'synthetic_sample',
    cadence_seconds: float = 45.0,
    scan_numbers=None,
    centre_ps: float = 2.25,
) -> THzData:
    """Build a THzData whose per-scan traces contain a KNOWN planted drift."""
    time_ps = start_ps + np.arange(n_time) * dt_ps
    relative_ps = time_ps - start_ps
    generator = np.random.default_rng(12345)
    first_timestamp = datetime.datetime(2026, 8, 20, 14, 24, 0)

    scans = []
    for index in range(n_scans):
        amplitude = 1.0 if amplitude_per_scan is None else amplitude_per_scan[index]
        delay = 0.0 if delay_ps_per_scan is None else delay_ps_per_scan[index]
        offset = 0.0 if offset_per_scan is None else offset_per_scan[index]
        trace = amplitude * _pulse(relative_ps - delay, centre_ps=centre_ps) + offset
        if noise:
            trace = trace + generator.normal(0.0, noise, size=n_time)
        number = index + 1 if scan_numbers is None else scan_numbers[index]
        timestamp = first_timestamp + datetime.timedelta(seconds=cadence_seconds * index)
        headers = [
            f"title {title} acc {number}",
            f"param Date and time,{timestamp.strftime('%Y-%m-%d %H:%M:%S')}.000000",
        ]
        scans.append(BaseTHzData(data=np.column_stack((time_ps, trace)), headers=headers))

    return THzData(data=scans, header=None, filename=f"{title}.acc", data_type='acc')


class _StubDataService:
    def __init__(self, data_dict):
        self.data_dict = data_dict


class _StubDataSet:
    """Minimal stand-in exposing only what extract_dataset_acquisitions consumes."""

    def __init__(self, data_dict, file_dir='', config=None):
        self.data = _StubDataService(data_dict)
        self.file_dir = file_dir
        self.config = config or {}


# ── header parsing ───────────────────────────────────────────────────────────


class TestHeaderParsing(unittest.TestCase):
    def test_scan_index_is_the_token_after_acc(self):
        """Regression: the scan number sits after 'acc', not at a fixed column."""
        scan = BaseTHzData(
            data=np.zeros((4, 2)),
            headers=['title sample_CNT_doped_1 acc 37',
                     'param Date and time,2026-08-20 14:24:00.000000'],
        )
        self.assertEqual(scan.filename, 'sample_CNT_doped_1')
        self.assertEqual(scan.scan_index, 37)

    def test_single_trace_header_has_no_scan_index(self):
        scan = BaseTHzData(
            data=np.zeros((4, 2)),
            headers=['title sample_CNT_doped_1',
                     'param Date and time,2026-08-20 14:24:00.000000'],
        )
        self.assertEqual(scan.filename, 'sample_CNT_doped_1')
        self.assertIsNone(scan.scan_index)

    def test_title_found_after_other_header_lines(self):
        """The whole header list is searched, not only its first entry."""
        scan = BaseTHzData(
            data=np.zeros((4, 2)),
            headers=['Created 3', 'type 0', 'title later_title acc 5',
                     'param Date and time,2026-08-20 14:24:00.000000'],
        )
        self.assertEqual(scan.filename, 'later_title')
        self.assertEqual(scan.scan_index, 5)

    def test_read_measurement_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sample_x.dat')
            with open(path, 'w') as handle:
                handle.write("%title sample_x\n%param accumulations,128\n"
                             "%param Lockin sensitivity,0.05\n"
                             "%param Reader function used,Read X\n"
                             "1.0 2.0\n%param ignored_after_data,9\n")
            parameters = tracking.read_measurement_parameters(path)
        self.assertEqual(parameters['accumulations'], 128.0)
        self.assertEqual(parameters['Lockin sensitivity'], 0.05)
        self.assertEqual(parameters['Reader function used'], 'Read X')
        self.assertNotIn('ignored_after_data', parameters)

    def test_find_measurement_parameters_prefers_the_dat_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, 'sample_x.dat'), 'w') as handle:
                handle.write("%param Lockin sensitivity,0.05\n1.0 2.0\n")
            with open(os.path.join(directory, 'sample_x.acc'), 'w') as handle:
                handle.write("%param Date and time,2026-08-20 14:24:00\n1.0 2.0\n")
            parameters = tracking.find_measurement_parameters(directory, 'sample_x.acc')
        self.assertEqual(parameters['Lockin sensitivity'], 0.05)
        self.assertEqual(parameters['source_header_file'], 'sample_x.dat')


# ── extraction ───────────────────────────────────────────────────────────────


class TestExtraction(unittest.TestCase):
    def test_shapes_axes_and_timestamps(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=12), n_fft=512)
        self.assertEqual(series.n_scans, 12)
        self.assertEqual(series.scan_traces.shape, (12, 135))
        self.assertEqual(series.scan_spectra.shape, (12, series.frequency_hz.size))
        self.assertEqual(series.frequency_hz.size, 512 // 2 + 1)
        # 45 s cadence over 11 intervals
        self.assertAlmostEqual(series.duration_seconds, 11 * 45.0)
        self.assertAlmostEqual(series.elapsed_seconds[0], 0.0)
        np.testing.assert_allclose(series.scan_numbers, np.arange(1, 13))
        # time axis converted from source ps to SI seconds
        self.assertAlmostEqual(series.time_axis_ps[0], 107.5, places=6)

    def test_baseline_subtraction_removes_a_constant_offset(self):
        """A per-scan DC offset must not survive into the spectra."""
        offsets = np.linspace(0.0, 0.02, 16)
        data_obj = _make_thzdata(n_scans=16, offset_per_scan=offsets)
        series = tracking.extract_acquisition_series(data_obj, baseline_ps=1.0, n_fft=512)
        # pre-pulse region should now sit at zero for every scan
        pre_pulse = series.scan_traces[:, :20].mean(axis=1)
        np.testing.assert_allclose(pre_pulse, 0.0, atol=1e-12)

    def test_offset_drift_does_not_masquerade_as_amplitude_drift(self):
        """The reason baseline subtraction exists: a wandering lock-in offset."""
        offsets = np.linspace(0.0, 0.05, 20)
        data_obj = _make_thzdata(n_scans=20, offset_per_scan=offsets)
        with_baseline = tracking.extract_acquisition_series(data_obj, baseline_ps=1.0, n_fft=512)
        ratio, _entry = tracking.compute_drift_metric(with_baseline, 'amplitude_ratio')
        index = with_baseline.frequency_index(0.5)
        np.testing.assert_allclose(ratio[:, index], 1.0, rtol=1e-6)

    def test_auto_half_width_fits_without_clipping(self):
        series = tracking.extract_acquisition_series(_make_thzdata(), half_width_ps=None, n_fft=512)
        self.assertTrue(series.processing['half_width_auto'])
        self.assertFalse(series.processing['window_clipped'])
        # pulse sits 2.25 ps into the record -> that is the widest symmetric fit
        self.assertAlmostEqual(series.processing['half_width_ps'], 2.25, places=6)

    def test_explicit_half_width_wider_than_the_record_is_flagged_clipped(self):
        series = tracking.extract_acquisition_series(_make_thzdata(), half_width_ps=5.0, n_fft=512)
        self.assertTrue(series.processing['window_clipped'])
        self.assertFalse(series.processing['half_width_auto'])

    def test_short_n_fft_raises_rather_than_truncating_the_pulse(self):
        with self.assertRaises(ValueError) as caught:
            tracking.extract_acquisition_series(_make_thzdata(), n_fft=64)
        self.assertIn('truncate', str(caught.exception))

    def test_ragged_acquisitions_raise(self):
        data_obj = _make_thzdata(n_scans=4)
        data_obj.data_list[2].raw_data = data_obj.data_list[2].raw_data[:-5]
        with self.assertRaises(ValueError) as caught:
            tracking.extract_acquisition_series(data_obj, n_fft=512)
        self.assertIn('do not share a time axis', str(caught.exception))

    def test_missing_acquisition_numbers_detected(self):
        numbers = [1, 2, 3, 7, 8]  # 4, 5, 6 culled
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=5, scan_numbers=numbers), n_fft=512
        )
        self.assertEqual(tracking.missing_acquisition_numbers(series), [4, 5, 6])

    def test_series_without_scans_raises_a_useful_message(self):
        class _NoScans:
            data_list = []
            filename = 'plain.dat'
        with self.assertRaises(ValueError) as caught:
            tracking.extract_acquisition_series(_NoScans(), n_fft=512)
        self.assertIn('.acc', str(caught.exception))


# ── drift metrics (known-answer) ─────────────────────────────────────────────


class TestDriftMetrics(unittest.TestCase):
    def test_amplitude_ratio_recovers_a_planted_gain_ramp(self):
        """Plant a linear 30% gain ramp; the metric must return exactly that."""
        n_scans = 40
        planted = np.linspace(1.0, 1.3, n_scans)
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=n_scans, amplitude_per_scan=planted), n_fft=512
        )
        ratio, entry = tracking.compute_drift_metric(series, 'amplitude_ratio', baseline_scans=4)
        self.assertEqual(entry.reference_value, 1.0)
        index = series.frequency_index(0.5)
        expected = planted / planted[:4].mean()
        np.testing.assert_allclose(ratio[:, index], expected, rtol=1e-8)

    def test_amplitude_ratio_is_flat_for_a_stable_measurement(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=20), n_fft=512)
        ratio, _entry = tracking.compute_drift_metric(series, 'amplitude_ratio')
        trusted = series.trusted_mask()
        np.testing.assert_allclose(ratio[:, trusted], 1.0, rtol=1e-8)

    def test_phase_deviation_recovers_a_planted_delay(self):
        """A planted time shift must appear as a phase ramp of slope -2*pi*tau.

        Uses a long record with a boxcar window wide enough to contain the pulse
        both before and after the shift.  With a Hann window the fixed window
        position (correctly) reshapes a shifted pulse, so the recovered delay is
        biased by the window rather than by the metric — a real effect, but not
        what this test is pinning down.
        """
        n_scans = 12
        delay_ps = np.zeros(n_scans)
        delay_ps[-1] = 0.10   # 100 fs on the final acquisition
        data_obj = _make_thzdata(
            n_scans=n_scans, delay_ps_per_scan=delay_ps,
            start_ps=0.0, n_time=400, centre_ps=10.0,
        )
        series = tracking.extract_acquisition_series(
            data_obj, half_width_ps=5.0, window_type='boxcar', n_fft=2048
        )
        phase, entry = tracking.compute_drift_metric(series, 'phase_deviation', baseline_scans=4)
        self.assertEqual(entry.reference_value, 0.0)

        # Fit the slope where the pulse has real amplitude and the wrapped phase
        # has not folded (|2*pi*f*tau| < pi  ->  f < 5 THz for tau = 0.1 ps).
        trusted = series.trusted_mask(snr_thresh_db=20.0)
        band = trusted & (series.frequency_thz > 0.2) & (series.frequency_thz < 2.0)
        slope, _intercept = np.polyfit(series.frequency_hz[band], phase[-1][band], 1)
        recovered_delay_s = -slope / (2 * np.pi)
        self.assertAlmostEqual(recovered_delay_s * 1e12, 0.10, places=3)

    def test_cumulative_average_equals_fft_of_the_running_mean_trace(self):
        """Frequency-domain shortcut must equal the time-domain definition exactly."""
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=16, amplitude_per_scan=np.linspace(1.0, 1.2, 16)),
            n_fft=512,
        )
        cumulative = tracking.cumulative_average_spectra(series)
        windowed = series.scan_traces * series.window_function
        for stop in (1, 5, 16):
            running_mean_trace = windowed[:stop].mean(axis=0)
            expected = np.fft.rfft(running_mean_trace, n=series.processing['n_fft'])
            np.testing.assert_allclose(cumulative[stop - 1], expected, rtol=1e-10, atol=1e-14)

    def test_cumulative_deviation_ends_at_zero_and_decreases(self):
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=30, noise=0.01), n_fft=512
        )
        deviation, entry = tracking.compute_drift_metric(series, 'cumulative_deviation')
        self.assertEqual(entry.reference_value, 0.0)
        self.assertFalse(entry.diverging)
        index = series.frequency_index(0.5)
        self.assertAlmostEqual(deviation[-1, index], 0.0, places=12)
        # averaging converges: the late running mean is closer than the early one
        self.assertLess(deviation[20, index], deviation[2, index])

    def test_cumulative_ratio_ends_at_one(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=20), n_fft=512)
        ratio, _entry = tracking.compute_drift_metric(series, 'cumulative_ratio')
        trusted = series.trusted_mask()
        np.testing.assert_allclose(ratio[-1, trusted], 1.0, rtol=1e-10)

    def test_baseline_spectrum_averages_the_requested_count(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=10), n_fft=512)
        expected = series.scan_spectra[:3].mean(axis=0)
        np.testing.assert_allclose(tracking.baseline_spectrum(series, 3), expected)
        # clamped to the available range rather than raising
        np.testing.assert_allclose(
            tracking.baseline_spectrum(series, 999), series.scan_spectra.mean(axis=0)
        )


# ── registry contract ────────────────────────────────────────────────────────


class TestMetricRegistry(unittest.TestCase):
    def test_the_shipped_metrics_are_registered_with_presentation_metadata(self):
        for name in ('amplitude_ratio', 'phase_deviation',
                     'cumulative_ratio', 'cumulative_deviation'):
            self.assertIn(name, tracking.DRIFT_METRICS)
            entry = tracking.DRIFT_METRICS[name]
            self.assertTrue(entry.label)
            self.assertTrue(entry.description)

    def test_unknown_metric_raises_and_lists_the_registered_names(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=4), n_fft=512)
        with self.assertRaises(KeyError) as caught:
            tracking.compute_drift_metric(series, 'not_a_metric')
        self.assertIn('amplitude_ratio', str(caught.exception))

    def test_options_are_filtered_to_each_metric_signature(self):
        """One option dict can be handed to every metric; each takes what it knows."""
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=8), n_fft=512)
        # cumulative_ratio takes no baseline_scans — this must not raise
        values, _entry = tracking.compute_drift_metric(
            series, 'cumulative_ratio', baseline_scans=4
        )
        self.assertEqual(values.shape, (8, series.n_frequencies))

    def test_registering_a_new_metric_makes_it_computable_and_plottable(self):
        @tracking.register_drift_metric(
            name='_test_metric', label='test', description='test only', reference_value=0.0
        )
        def _test_metric(series):
            return np.zeros((series.n_scans, series.n_frequencies))

        try:
            self.assertIn('_test_metric', tracking.available_drift_metrics())
            series = tracking.extract_acquisition_series(_make_thzdata(n_scans=6), n_fft=512)
            values, entry = tracking.compute_drift_metric(series, '_test_metric')
            self.assertEqual(entry.label, 'test')
            np.testing.assert_allclose(values, 0.0)
            ax = tracking.plot_drift_map(series, '_test_metric')
            self.assertEqual(ax.get_ylabel(), 'Frequency (THz)')
            plt.close('all')
        finally:
            tracking.DRIFT_METRICS.pop('_test_metric', None)


# ── timing drift ─────────────────────────────────────────────────────────────


class TestTimingDrift(unittest.TestCase):
    """The sub-sample delay measurement, which is the point of the phase metric."""

    def _ramped_series(self, total_delay_ps: float, n_scans: int = 40):
        delays = np.linspace(0.0, total_delay_ps, n_scans)
        data_obj = _make_thzdata(
            n_scans=n_scans, delay_ps_per_scan=delays,
            start_ps=0.0, n_time=400, centre_ps=10.0,
        )
        return tracking.extract_acquisition_series(
            data_obj, half_width_ps=5.0, window_type='boxcar', n_fft=2048
        )

    def test_recovers_a_planted_delay_ramp(self):
        series = self._ramped_series(total_delay_ps=0.040)   # 40 fs over the run
        delays_fs = tracking.fitted_delay_seconds(series, baseline_scans=1) * 1e15
        expected_fs = np.linspace(0.0, 40.0, series.n_scans)
        np.testing.assert_allclose(delays_fs, expected_fs, atol=0.5)

    def test_resolves_drift_far_below_the_sample_step(self):
        """30 fs of smooth drift on a 50 fs grid.

        The time-domain peak cannot represent it: it sits on one bin, then jumps a
        whole 50 fs step once the pulse crosses the half-bin boundary — reporting
        either 0 or 50 fs when the truth is a smooth ramp to 30. The phase fit
        recovers the ramp itself.
        """
        series = self._ramped_series(total_delay_ps=0.030)
        peak_shift_fs = (series.peak_time_seconds_per_scan
                         - series.peak_time_seconds_per_scan[0]) * 1e15
        # only ever two values, one sample step apart — a staircase, not a ramp
        unique_shifts = np.unique(np.round(peak_shift_fs, 6))
        np.testing.assert_allclose(unique_shifts, [0.0, 50.0], atol=1e-6)

        result = tracking.delay_drift_rate(series, baseline_scans=1)
        self.assertAlmostEqual(result['total_seconds'] * 1e15, 27.0, delta=2.0)
        # and the trend is far above the scan-to-scan noise floor of the fit
        self.assertGreater(abs(result['total_seconds']), 10 * result['scatter_seconds'])

    def test_rate_per_hour_matches_the_planted_ramp(self):
        series = self._ramped_series(total_delay_ps=0.040)
        result = tracking.delay_drift_rate(series, baseline_scans=1)
        expected_rate_fs = 40.0 / series.elapsed_hours[-1]
        self.assertAlmostEqual(result['rate_seconds_per_hour'] * 1e15,
                               expected_rate_fs, delta=expected_rate_fs * 0.02)

    def test_stable_measurement_has_no_delay_drift(self):
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=20, start_ps=0.0, n_time=400, centre_ps=10.0),
            half_width_ps=5.0, n_fft=2048,
        )
        result = tracking.delay_drift_rate(series)
        self.assertAlmostEqual(result['total_seconds'] * 1e15, 0.0, places=6)

    def test_too_few_trusted_bins_raises_a_useful_error(self):
        series = self._ramped_series(total_delay_ps=0.02, n_scans=6)
        with self.assertRaises(ValueError) as caught:
            tracking.fitted_delay_seconds(series, band_thz=(40.0, 45.0))
        self.assertIn('band_thz', str(caught.exception))

    def test_timing_drift_figure_builds(self):
        series = self._ramped_series(total_delay_ps=0.03, n_scans=12)
        ax = tracking.plot_timing_drift(series, baseline_scans=1)
        self.assertIn('Delay', ax.get_ylabel())
        plt.close('all')


# ── purge equilibration ──────────────────────────────────────────────────────


class TestEquilibrationFit(unittest.TestCase):
    """The saturating-exponential fit and, more importantly, its verdict."""

    def test_recovers_a_planted_time_constant(self):
        seconds = np.linspace(0, 7800, 174)
        values = 1.2 - 0.2 * np.exp(-seconds / (40 * 60))
        fit = tracking.fit_exponential_equilibration(seconds, values)
        self.assertTrue(fit['transient_detected'])
        self.assertAlmostEqual(fit['tau_seconds'] / 60, 40.0, places=3)
        self.assertAlmostEqual(fit['plateau'], 1.2, places=6)

    def test_flat_series_reports_no_transient(self):
        seconds = np.linspace(0, 1800, 43)
        generator = np.random.default_rng(7)
        values = 1.0 + generator.normal(0, 1e-4, seconds.size)
        fit = tracking.fit_exponential_equilibration(seconds, values)
        self.assertFalse(fit['transient_detected'])
        self.assertTrue(fit['reason'])

    def test_pure_straight_line_is_not_called_a_transient(self):
        """A linear ramp has no plateau; the fit must not claim one."""
        seconds = np.linspace(0, 7800, 100)
        fit = tracking.fit_exponential_equilibration(seconds, 1.0 + 3e-5 * seconds)
        self.assertFalse(fit['transient_detected'])

    def test_noise_estimate_rejects_a_fit_to_scatter(self):
        seconds = np.linspace(0, 1800, 40)
        values = 1.0 - 0.001 * np.exp(-seconds / 300.0)
        # a generous noise estimate must veto this small a span
        fit = tracking.fit_exponential_equilibration(seconds, values, noise_estimate=0.01)
        self.assertFalse(fit['transient_detected'])
        self.assertIn('noise', fit['reason'])

    def test_too_few_points(self):
        fit = tracking.fit_exponential_equilibration([0.0, 1.0], [1.0, 1.0])
        self.assertFalse(fit['transient_detected'])
        self.assertIn('fewer than 4', fit['reason'])

    def test_time_to_equilibrate_matches_the_exponential(self):
        self.assertAlmostEqual(tracking.time_to_equilibrate(60.0, 0.63), 60.0, delta=0.6)
        self.assertAlmostEqual(tracking.time_to_equilibrate(60.0, 0.99),
                               60.0 * -np.log(0.01), places=6)
        self.assertTrue(np.isnan(tracking.time_to_equilibrate(-1.0)))

    def test_report_detects_a_planted_purge_transient(self):
        """Amplitude rising to a plateau, as a purge does — must be caught."""
        n_scans = 100
        seconds = np.arange(n_scans) * 45.0
        gain = 1.2 - 0.2 * np.exp(-seconds / (40 * 60))
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=n_scans, amplitude_per_scan=gain,
                          start_ps=0.0, n_time=400, centre_ps=10.0),
            half_width_ps=5.0, n_fft=1024,
        )
        report = tracking.purge_equilibration_report(series, frequencies_thz=(0.5, 1.0))
        amplitude_fits = list(report['amplitude'].values())
        self.assertTrue(any(fit['transient_detected'] for fit in amplitude_fits))
        detected = [fit for fit in amplitude_fits if fit['transient_detected']]
        for fit in detected:
            self.assertAlmostEqual(fit['tau_seconds'] / 60, 40.0, delta=3.0)
        self.assertIn('PURGE TRANSIENT', report['verdict'])

    def test_fixed_tau_fits_only_plateau_and_span(self):
        seconds = np.linspace(0, 3000, 60)
        values = 1.2 - 0.2 * np.exp(-seconds / (40 * 60))
        fit = tracking.fit_exponential_equilibration(
            seconds, values, fixed_tau_seconds=40 * 60
        )
        self.assertTrue(fit['tau_fixed'])
        self.assertAlmostEqual(fit['tau_seconds'], 40 * 60, places=6)
        self.assertAlmostEqual(fit['plateau'], 1.2, places=6)
        self.assertTrue(fit['transient_detected'])

    def test_fixed_tau_rescues_a_short_run_sitting_on_a_slow_tail(self):
        """The reason fixed tau exists: a short record cannot fit its own tau.

        A window taken LATE on a transient has curvature too weak to beat a straight
        line *once realistic noise is present* — fitting freely calls it settled.
        Supplying the tau measured on the long run recovers the residual transient.
        The noise matters: on noiseless data even a trace of curvature fits
        perfectly, so this failure mode only appears with real scatter.
        """
        tau = 45 * 60.0
        late_start = 2.8 * tau
        seconds = np.linspace(0, 1900, 43)
        clean = 1.2 - 0.2 * np.exp(-(late_start + seconds) / tau)
        noisy = clean + np.random.default_rng(3).normal(0, 0.002, seconds.size)

        free_fit = tracking.fit_exponential_equilibration(seconds, noisy)
        self.assertFalse(free_fit['transient_detected'])   # looks flat/linear

        fixed_fit = tracking.fit_exponential_equilibration(
            seconds, noisy, fixed_tau_seconds=tau
        )
        self.assertTrue(fixed_fit['transient_detected'])
        self.assertAlmostEqual(fixed_fit['plateau'], 1.2, places=2)

    def test_report_clears_a_stable_run(self):
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=43, start_ps=0.0, n_time=400, centre_ps=10.0),
            half_width_ps=5.0, n_fft=1024,
        )
        report = tracking.purge_equilibration_report(series, frequencies_thz=(0.5, 1.0))
        self.assertIn('EQUILIBRATED', report['verdict'])


class TestSettlingMetric(unittest.TestCase):
    """The residual-drift criterion and the cut index it produces."""

    @staticmethod
    def _fit(span=0.2, tau_minutes=45.0):
        return dict(span=span, tau_seconds=tau_minutes * 60.0,
                    plateau=1.0 + span, transient_detected=True)

    def test_residual_drift_matches_the_closed_form(self):
        fit = self._fit()
        tau, window = fit['tau_seconds'], 1800.0
        for start in (0.0, 1200.0, 5000.0):
            expected = 0.2 * np.exp(-start / tau) * (1 - np.exp(-window / tau))
            self.assertAlmostEqual(
                tracking.residual_drift_over_window(fit, start, window), expected, places=12
            )

    def test_drift_decays_with_start_time_and_grows_with_window(self):
        fit = self._fit()
        self.assertGreater(tracking.residual_drift_over_window(fit, 0, 1800),
                           tracking.residual_drift_over_window(fit, 3600, 1800))
        self.assertGreater(tracking.residual_drift_over_window(fit, 0, 3600),
                           tracking.residual_drift_over_window(fit, 0, 1800))

    def test_wait_time_inverts_the_drift_formula(self):
        """Round trip: wait the returned time, and the drift equals the threshold."""
        fit = self._fit()
        window, threshold = 1800.0, 0.005
        wait = tracking.time_until_acceptable_drift(fit, window, threshold)
        self.assertGreater(wait, 0)
        self.assertAlmostEqual(
            tracking.residual_drift_over_window(fit, wait, window), threshold, places=12
        )

    def test_no_wait_when_already_acceptable(self):
        fit = self._fit(span=1e-5)
        self.assertEqual(tracking.time_until_acceptable_drift(fit, 1800.0, 0.005), 0.0)

    def test_no_transient_means_no_drift_and_no_wait(self):
        fit = dict(span=0.2, tau_seconds=float('nan'), plateau=1.2)
        self.assertEqual(tracking.residual_drift_over_window(fit, 0, 1800), 0.0)
        self.assertEqual(tracking.time_until_acceptable_drift(fit, 1800, 0.005), 0.0)

    def _transient_series(self, n_scans=250, tau_minutes=45.0, span=0.2, title='transient'):
        """A run long enough to actually reach the criterion.

        Note 250 scans x 45 s = 187 min against a 45 min tau. That is not padding:
        with a 30 min window and a 0.5% threshold the criterion is not met until
        ~134 min, so a shorter run genuinely never becomes acceptable.
        """
        seconds = np.arange(n_scans) * 45.0
        gain = (1.0 + span) - span * np.exp(-seconds / (tau_minutes * 60.0))
        return tracking.extract_acquisition_series(
            _make_thzdata(n_scans=n_scans, amplitude_per_scan=gain,
                          start_ps=0.0, n_time=400, centre_ps=10.0, title=title),
            half_width_ps=5.0, n_fft=1024,
        )

    def test_cut_index_is_where_the_criterion_first_holds(self):
        series = self._transient_series()
        assessment = tracking.purge_settling_assessment(
            series, frequency_thz=1.0, measurement_window_minutes=30.0,
            acceptable_drift_fraction=0.005,
        )
        cut = assessment['first_acceptable_index']
        self.assertIsNotNone(cut)
        fit = assessment['fit']
        window = 30.0 * 60.0
        # acceptable at the cut, and NOT acceptable one acquisition earlier
        self.assertLessEqual(
            tracking.residual_drift_over_window(fit, series.elapsed_seconds[cut], window),
            0.005 + 1e-9,
        )
        self.assertGreater(
            tracking.residual_drift_over_window(fit, series.elapsed_seconds[cut - 1], window),
            0.005,
        )

    def test_a_tighter_threshold_pushes_the_cut_later(self):
        series = self._transient_series()
        loose = tracking.purge_settling_assessment(
            series, frequency_thz=1.0, acceptable_drift_fraction=0.02)
        tight = tracking.purge_settling_assessment(
            series, frequency_thz=1.0, acceptable_drift_fraction=0.002)
        self.assertGreater(tight['first_acceptable_index'], loose['first_acceptable_index'])

    def test_stable_run_is_acceptable_from_the_first_acquisition(self):
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=40, start_ps=0.0, n_time=400, centre_ps=10.0),
            half_width_ps=5.0, n_fft=1024,
        )
        assessment = tracking.purge_settling_assessment(series, frequency_thz=1.0)
        self.assertEqual(assessment['first_acceptable_index'], 0)
        self.assertIn('EQUILIBRATED', assessment['verdict'])

    def test_run_too_short_to_ever_become_acceptable(self):
        series = self._transient_series(n_scans=30, tau_minutes=45.0, span=0.3)
        assessment = tracking.purge_settling_assessment(
            series, frequency_thz=1.0, acceptable_drift_fraction=0.001)
        self.assertIsNone(assessment['first_acceptable_index'])
        self.assertIn('NOT YET ACCEPTABLE', assessment['verdict'])

    def test_record_tau_ratio_flags_an_extrapolated_plateau(self):
        series = self._transient_series(n_scans=140, tau_minutes=45.0)
        assessment = tracking.purge_settling_assessment(series, frequency_thz=1.0)
        # 140 scans x 45 s = 105 min over a 45 min tau
        self.assertAlmostEqual(assessment['record_tau_ratio'], 105.0 / 45.0, delta=0.3)

    def test_known_tau_changes_the_verdict_on_a_short_tail_run(self):
        """A short late-window run: free fit says clean, supplied tau says not yet."""
        tau_minutes, span = 45.0, 0.2
        n_scans = 40
        late_seconds = 2.8 * tau_minutes * 60.0
        seconds = np.arange(n_scans) * 45.0
        gain = (1.0 + span) - span * np.exp(-(late_seconds + seconds) / (tau_minutes * 60))
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=n_scans, amplitude_per_scan=gain,
                          start_ps=0.0, n_time=400, centre_ps=10.0, noise=0.004),
            half_width_ps=5.0, n_fft=1024,
        )
        naive = tracking.purge_settling_assessment(series, frequency_thz=1.0)
        informed = tracking.purge_settling_assessment(
            series, frequency_thz=1.0, known_tau_seconds=tau_minutes * 60.0)
        self.assertIn('EQUILIBRATED', naive['verdict'])
        self.assertTrue(informed['fit']['tau_fixed'])
        self.assertGreater(informed['residual_drift'], 0.0)

    def test_print_settling_assessment_runs_for_both_verdicts(self):
        transient = tracking.purge_settling_assessment(
            self._transient_series(), frequency_thz=1.0)
        stable = tracking.purge_settling_assessment(
            tracking.extract_acquisition_series(
                _make_thzdata(n_scans=30, start_ps=0.0, n_time=400, centre_ps=10.0),
                half_width_ps=5.0, n_fft=1024),
            frequency_thz=1.0)
        for assessment in (transient, stable):
            self.assertIs(tracking.print_settling_assessment(assessment), assessment)


class TestPurgeFigure(unittest.TestCase):
    def _series(self, n_scans=80, span=0.2, tau_minutes=45.0, title='transient_run'):
        seconds = np.arange(n_scans) * 45.0
        gain = (1.0 + span) - span * np.exp(-seconds / (tau_minutes * 60.0))
        return tracking.extract_acquisition_series(
            _make_thzdata(n_scans=n_scans, amplitude_per_scan=gain,
                          start_ps=0.0, n_time=400, centre_ps=10.0, title=title),
            half_width_ps=5.0, n_fft=1024,
        )

    def tearDown(self):
        plt.close('all')

    def test_single_series_figure_has_fit_and_residual_panels(self):
        figure, axes, fits = tracking.plot_purge_equilibration(
            self._series(), frequencies_thz=(1.0,), include_timing=False)
        self.assertEqual(axes.shape, (2, 1))
        self.assertIn('Residual (%)', axes[1, 0].get_ylabel())
        self.assertTrue(any(fit['transient_detected'] for fit in list(fits.values())[0].values()))

    def test_timing_panel_adds_a_third_row(self):
        figure, axes, _fits = tracking.plot_purge_equilibration(
            self._series(), frequencies_thz=(1.0,), include_timing=True)
        self.assertEqual(axes.shape, (3, 1))
        self.assertIn('Delay', axes[2, 0].get_ylabel())

    def test_control_adds_a_second_column_with_shared_row_limits(self):
        control = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=40, start_ps=0.0, n_time=400, centre_ps=10.0,
                          title='control_run'),
            half_width_ps=5.0, n_fft=1024,
        )
        figure, axes, fits = tracking.plot_purge_equilibration(
            self._series(), frequencies_thz=(1.0,), control_series=control)
        self.assertEqual(axes.shape, (3, 2))
        self.assertEqual(len(fits), 2)
        for row in range(3):
            self.assertEqual(axes[row, 0].get_ylim(), axes[row, 1].get_ylim())

    def test_assessment_draws_the_cut_marker(self):
        series = self._series()
        # 0.05 so the criterion is reached inside this 60 min run (see the
        # _transient_series note: a tight threshold needs a much longer record).
        assessment = tracking.purge_settling_assessment(
            series, frequency_thz=1.0, acceptable_drift_fraction=0.05)
        self.assertIsNotNone(assessment['first_acceptable_minutes'])
        figure, axes, _fits = tracking.plot_purge_equilibration(
            series, frequencies_thz=(1.0,), include_timing=False, assessment=assessment)
        cut_lines = [line for line in axes[0, 0].get_lines()
                     if len(line.get_xdata()) == 2
                     and line.get_xdata()[0] == line.get_xdata()[1]]
        self.assertTrue(cut_lines, 'expected a vertical cut marker')

    def test_control_tau_can_be_fixed_from_the_main_run(self):
        control = self._series(n_scans=30, span=0.2, title='control_run')
        figure, axes, fits = tracking.plot_purge_equilibration(
            self._series(), frequencies_thz=(1.0,), control_series=control,
            control_fixed_tau_seconds=45 * 60.0, include_timing=False)
        control_fits = fits[control.filename]
        self.assertTrue(any(fit['tau_fixed'] for fit in control_fits.values()))


# ── summary and export ───────────────────────────────────────────────────────


class TestSummaryAndExport(unittest.TestCase):
    def test_drift_rate_recovers_a_planted_per_hour_slope(self):
        """40 acquisitions at 45 s = 0.4875 h; a 1.0 -> 1.3 ramp is +0.6154 /hour."""
        n_scans = 40
        planted = np.linspace(1.0, 1.3, n_scans)
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=n_scans, amplitude_per_scan=planted), n_fft=512
        )
        rates = tracking.drift_rate_per_hour(
            series, [0.5], metric='amplitude_ratio', baseline_scans=1
        )
        result = list(rates.values())[0]
        expected_slope = (planted[-1] - planted[0]) / planted[0] / series.elapsed_hours[-1]
        self.assertAlmostEqual(result['slope_per_hour'], expected_slope, places=6)

    def test_summarise_drift_returns_rates_for_each_frequency(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=20), n_fft=512)
        rates = tracking.summarise_drift(series, frequencies_thz=(0.5, 1.0))
        self.assertEqual(len(rates), 2)

    def test_save_metric_table_writes_a_readable_csv(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=10), n_fft=512)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'drift.csv')
            tracking.save_metric_table(series, path, frequencies_thz=(0.5, 1.0))
            with open(path) as handle:
                header = handle.readline().strip().split(',')
            table = np.loadtxt(path, delimiter=',', skiprows=1)
        self.assertEqual(header[:2], ['elapsed_minutes', 'acquisition_number'])
        self.assertEqual(len(header), 4)
        self.assertEqual(table.shape, (10, 4))

    def test_describe_reports_the_processing_actually_applied(self):
        series = tracking.extract_acquisition_series(
            _make_thzdata(n_scans=6), half_width_ps=1.5, n_fft=512
        )
        text = series.describe()
        self.assertIn('acquisitions', text)
        self.assertIn('1.50 ps', text)


# ── headless workflow ────────────────────────────────────────────────────────


class TestDriftWorkflow(unittest.TestCase):
    """Full script path on an Agg backend, as a user would run it."""

    def test_dataset_extraction_and_every_figure(self):
        sample = _make_thzdata(
            n_scans=30,
            amplitude_per_scan=np.linspace(1.0, 1.15, 30),
            title='sample_7_CNT-doped-1',
        )
        reference = _make_thzdata(
            n_scans=24, start_ps=113.0, n_time=141, title='reference_7_gold_2'
        )
        dataset = _StubDataSet(
            {'sample_7_CNT-doped-1.acc': sample, 'reference_7_gold_2.acc': reference},
            config={
                'acquisition_tracking': {'baseline_ps': 1.0, 'baseline_scans': 4},
                'window': {'type': 'hann', 'alpha': 1.0},
                'fft': {'n_fft': 1024},
            },
        )

        series_by_filename = tracking.extract_dataset_acquisitions(dataset)
        self.assertEqual(set(series_by_filename), {'sample_7_CNT-doped-1.acc',
                                                   'reference_7_gold_2.acc'})
        # Different record lengths, one shared frequency grid (same dt, same n_fft).
        grids = [series.frequency_hz for series in series_by_filename.values()]
        np.testing.assert_allclose(grids[0], grids[1])

        for series in series_by_filename.values():
            tracking.summarise_drift(series, frequencies_thz=(0.5, 1.0))
            tracking.show_drift_figures(
                series,
                frequencies_thz=(0.5, 1.0),
                metrics=('amplitude_ratio', 'cumulative_deviation', 'phase_deviation'),
                baseline_scans=4,
            )
        tracking.plot_drift_comparison(series_by_filename, frequency_thz=0.5)
        self.assertGreater(len(plt.get_fignums()), 0)
        plt.close('all')

    def test_file_selector_limits_the_extraction(self):
        dataset = _StubDataSet({
            'sample_a.acc': _make_thzdata(n_scans=6, title='sample_a'),
            'reference_b.acc': _make_thzdata(n_scans=6, title='reference_b'),
        })
        selected = tracking.extract_dataset_acquisitions(dataset, files='sample')
        self.assertEqual(list(selected), ['sample_a.acc'])

    def test_files_without_per_scan_data_are_skipped_not_fatal(self):
        class _NoScans:
            data_list = []
            filename = 'plain.dat'
        dataset = _StubDataSet({
            'sample_a.acc': _make_thzdata(n_scans=6, title='sample_a'),
            'plain.dat': _NoScans(),
        })
        extracted = tracking.extract_dataset_acquisitions(dataset)
        self.assertEqual(list(extracted), ['sample_a.acc'])

    def test_plot_raises_a_useful_error_when_no_band_survives_the_mask(self):
        series = tracking.extract_acquisition_series(_make_thzdata(n_scans=6), n_fft=512)
        with self.assertRaises(ValueError) as caught:
            tracking.plot_drift_map(
                series, plot_config={'frequency_range_thz': (40.0, 50.0)}
            )
        self.assertIn('show_mask', str(caught.exception))
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
