"""Inspection of every processing step: the data each figure needs, the figures, and when to show them.

An ellipsometry run is a chain of steps (``stages.CHECKPOINTS``): the raw traces, the windowing,
the spectra, the harmonic fit, the channel calibration, and the result. Each step has figures,
registered here with ``@inspection_figure("<checkpoint>")``; the registry is the single source
of truth for which figures exist, at which step, and what they are called. Three things use it:

``StepwiseInspector``        an observer for ``run_ellipsometry``: shows each step's figures as
                             the run passes it (``config['general']['inspect']``).
``save_inspection``          writes ``inspection.npz`` + ``figures/*.png`` into a run's bundle.
``thz_ellipsometry_view.py`` loads a bundle's ``inspection.npz`` and shows any step again,
                             without rerunning anything.

Figures never read a ``RunOutcome`` or a file: they draw an :class:`InspectionRecord`, plain
arrays built from the run's state by :func:`build_inspection_record`, so a figure drawn live and
one drawn from a bundle months later are the same figure. The drawing is in
``inspection_plots``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

import numpy as np

from ..core import materials
from ..core.model import ellipsometric_ratio
from .loader import SAMPLE_ROLE

__all__ = [
    "INSPECTION_FIGURES",
    "InspectionRecord",
    "SeriesInspection",
    "StepwiseInspector",
    "build_inspection_record",
    "inspection_figure",
    "load_inspection",
    "make_inspection_figures",
    "save_inspection",
]

CHANNEL_REFERENCE_ROLE = "channel_reference"
INSPECTION_FILENAME = "inspection.npz"
FIGURE_DIRECTORY = "figures"


# ---------------------------------------------------------------------------
# The data the figures draw
# ---------------------------------------------------------------------------

@dataclass
class SeriesInspection:
    """One fitted series (a role at a probe setting), as plain arrays. None = not reached yet."""

    role: str
    probe_deg: float
    label: str
    # raw_traces
    time_ps: np.ndarray
    row_traces: np.ndarray                 #: (n_rows, n_samples) as loaded, before the baseline
    row_angles_deg: np.ndarray             #: (n_rows,) THz polarisation of each row
    row_elapsed_s: np.ndarray | None = None
    row_segment_ids: np.ndarray | None = None
    baseline_samples: int = 0              #: leading samples whose mean is the baseline
    # windowing
    window: np.ndarray | None = None       #: (n_samples,) total weighting (window x edge taper)
    window_centre_index: int = -1
    window_half_width_samples: int = -1
    window_clipped_before: int = 0
    window_clipped_after: int = 0
    window_shape: str = ""
    taper_fraction: float = np.nan
    edge_taper_ps: float = np.nan
    # spectra
    frequencies_hz: np.ndarray | None = None
    row_spectra: np.ndarray | None = None  #: (n_rows, n_frequencies) complex
    row_variance: np.ndarray | None = None  #: (n_rows, n_frequencies) E|dS|^2 from the noise model
    resolution_df_hz: float = np.nan
    oversampling: float = np.nan
    # harmonic_fit
    channel_p: np.ndarray | None = None
    channel_s: np.ndarray | None = None
    background: np.ndarray | None = None
    variance_p: np.ndarray | None = None
    variance_s: np.ndarray | None = None
    delays_s: np.ndarray | None = None     #: (n_rows,) fitted delay of each row
    row_scales: np.ndarray | None = None   #: (n_rows, n_frequencies) fitted scale of each row
    predicted_spectra: np.ndarray | None = None   #: (n_rows, n_frequencies) the model
    drift_model: str = ""
    amplitude_model: str = ""
    reduced_chi_square: float = np.nan
    nuisance_names: np.ndarray | None = None
    nuisance_values: np.ndarray | None = None
    nuisance_errors: np.ndarray | None = None

    @property
    def key(self):
        return (self.role, self.probe_deg)

    @property
    def baseline_corrected_traces(self):
        return self.row_traces - self.row_traces[:, :max(self.baseline_samples, 1)].mean(
            axis=1, keepdims=True)

    @property
    def states_deg(self):
        """The distinct polarisation states, in the order first measured."""
        _, first = np.unique(np.round(self.row_angles_deg % 360.0, 3), return_index=True)
        return np.round(self.row_angles_deg % 360.0, 3)[np.sort(first)]

    def rows_in_state(self, state_deg):
        return np.isclose(np.round(self.row_angles_deg % 360.0, 3), state_deg)


@dataclass
class InspectionRecord:
    """Everything the inspection figures draw, for one run (or the part of it run so far)."""

    series: list = field(default_factory=list)       #: SeriesInspection, reference first
    primary_probe_deg: float = np.nan
    # harmonic_fit (the band is chosen right after the fits)
    band: np.ndarray | None = None
    independent_band_points: np.ndarray | None = None   #: over the band
    # calibration
    incidence_angle_deg: float = np.nan
    calibration_name: str = ""
    calibration_per_frequency: np.ndarray | None = None  #: (n_frequencies,) from its source
    calibration_applied: complex = np.nan                 #: the value divided out
    #: name -> (n_frequencies,) complex: C implied by each series of KNOWN material (gold
    #: reference, a validated sample), side by side
    calibration_by_material: dict = field(default_factory=dict)
    # result
    band_frequencies_hz: np.ndarray | None = None
    ratio: np.ndarray | None = None          #: rho over the band
    index: np.ndarray | None = None          #: N = n - ik over the band
    index_standard_error: np.ndarray | None = None
    reference_material: str = ""             #: what the sample should be, if known
    reference_index: np.ndarray | None = None
    run_report: str = ""

    def series_for(self, role):
        return [entry for entry in self.series if entry.role == role]


def _series_order(key):
    role, probe = key
    return (0 if role == CHANNEL_REFERENCE_ROLE else 1 if role == SAMPLE_ROLE else 2, role,
            -1.0 if probe is None else probe)


def _known_material(config):
    validation = (config or {}).get("validation") or {}
    for name in (validation.get("expect"), validation.get("cross_check_material")):
        if name in materials.REFERENCE_MATERIALS:
            return name
    return ""


def _series_inspection(key, rows, config, transformed=None, noise=None, fit=None):
    preprocess = (config or {}).get("preprocess") or {}
    role, probe = key
    entry = SeriesInspection(
        role=role, probe_deg=np.nan if probe is None else float(probe),
        label=f"{role} @ probe {probe:g} deg" if probe is not None else role,
        time_ps=np.asarray(rows.time_ps, dtype=float),
        row_traces=np.asarray(rows.traces, dtype=float),
        row_angles_deg=np.asarray(rows.angles_deg, dtype=float),
        row_elapsed_s=None if rows.elapsed_seconds is None else np.asarray(rows.elapsed_seconds),
        row_segment_ids=None if rows.segment_ids is None else np.asarray(rows.segment_ids),
        baseline_samples=max(int(preprocess.get("baseline_fraction", 0.1)
                                 * np.size(rows.time_ps)), 1),
        window_shape=str(preprocess.get("window_shape", "tukey")),
        taper_fraction=float(preprocess.get("taper_fraction", 0.5)),
        edge_taper_ps=float(preprocess.get("edge_taper_ps", 0.5)))
    if transformed is not None:
        entry.window = transformed.window
        placement = transformed.window_placement
        if placement is not None:
            entry.window_centre_index = placement.centre_index
            entry.window_half_width_samples = placement.half_width_samples
            entry.window_clipped_before = placement.clipped_before
            entry.window_clipped_after = placement.clipped_after
        entry.frequencies_hz = transformed.frequencies_hz
        entry.row_spectra = transformed.spectra
        if transformed.resolution is not None:
            entry.resolution_df_hz = transformed.resolution.df_resolution_hz
            entry.oversampling = transformed.resolution.oversampling
    if noise is not None and noise.spectral_variance is not None:
        entry.row_variance = noise.spectral_variance
    if fit is not None:
        entry.channel_p, entry.channel_s, entry.background = (fit.channel_p, fit.channel_s,
                                                              fit.background)
        if fit.covariance is not None:
            entry.variance_p = fit.covariance[:, 0, 0].real
            entry.variance_s = fit.covariance[:, 1, 1].real
        entry.delays_s = fit.delays_s
        entry.row_scales = fit.row_scales
        entry.predicted_spectra = fit.predicted_spectra(transformed.frequencies_hz)
        entry.drift_model, entry.amplitude_model = fit.drift_model, fit.amplitude_model
        entry.reduced_chi_square = (np.nan if fit.reduced_chi_square is None
                                    else float(fit.reduced_chi_square))
        if fit.nuisance_parameters is not None:
            entry.nuisance_names = np.array(fit.nuisance_parameter_names)
            entry.nuisance_values = fit.nuisance_parameters
            entry.nuisance_errors = fit.nuisance_parameter_errors
    return entry


def build_inspection_record(state):
    """An InspectionRecord from a run's state -- partial (a checkpoint's) or complete.

    ``state`` is the dict an observer receives (``stages.CHECKPOINTS``); use
    :func:`record_from_outcome` for a finished RunOutcome.
    """
    config = state.get("config") or {}
    rows = state.get("rows") or {}
    transformed = state.get("transformed") or {}
    noise = state.get("noise") or {}
    fits = state.get("fits") or {}
    record = InspectionRecord(
        series=[_series_inspection(key, rows[key], config, transformed.get(key),
                                   noise.get(key), fits.get(key))
                for key in sorted(rows, key=_series_order)],
        primary_probe_deg=(np.nan if state.get("primary_probe_deg") is None
                           else float(state["primary_probe_deg"])))

    if state.get("band") is not None:
        record.band = np.asarray(state["band"], dtype=bool)
        outcome = state.get("outcome")
        if outcome is not None:
            record.independent_band_points = outcome.independent_band_points()
        else:
            from thz_core.thz_core import conditioning
            sample = transformed.get((SAMPLE_ROLE, state.get("primary_probe_deg")))
            factor = (1 if sample is None or sample.resolution is None
                      else sample.resolution.decimation_factor)
            count = int(record.band.sum())
            record.independent_band_points = np.zeros(count, dtype=bool)
            record.independent_band_points[conditioning.independent_bin_indices(
                np.ones(count, dtype=bool), factor)] = True

    calibration = state.get("calibration")
    if calibration is not None:
        angle = float(state.get("incidence_angle_rad"))
        record.incidence_angle_deg = float(np.rad2deg(angle))
        record.calibration_name = str(calibration.reference_name)
        record.calibration_per_frequency = calibration.ratio_per_frequency
        record.calibration_applied = complex(calibration.ratio)
        frequencies = state["frequencies_hz"]
        reference_fit = fits.get((CHANNEL_REFERENCE_ROLE, state.get("primary_probe_deg")))
        if reference_fit is not None:
            record.calibration_by_material["gold reference"] = (
                reference_fit.channel_ratio
                / ellipsometric_ratio(materials.gold_index(frequencies), angle, 1.0))
        material = _known_material(config)
        sample_fit = fits.get((SAMPLE_ROLE, state.get("primary_probe_deg")))
        if material and sample_fit is not None:
            record.reference_material = material
            record.calibration_by_material[f"sample as {material}"] = (
                sample_fit.channel_ratio / ellipsometric_ratio(
                    materials.reference_index(material, frequencies), angle, 1.0))

    result = state.get("result")
    if result is not None:
        inversion = result.inversion
        record.band_frequencies_hz = inversion.frequencies_hz
        record.ratio = inversion.ratio
        record.index = inversion.index
        record.index_standard_error = result.index_standard_error
        if record.reference_material:
            record.reference_index = materials.reference_index(record.reference_material,
                                                               inversion.frequencies_hz)
        outcome = state.get("outcome")
        if outcome is not None:
            from .report import format_run_report
            record.run_report = format_run_report(outcome)
    return record


def record_from_outcome(outcome):
    """The complete InspectionRecord of a finished run."""
    result = outcome.result
    return build_inspection_record({
        "config": outcome.config, "rows": outcome.rows, "transformed": outcome.transformed,
        "noise": outcome.noise, "fits": outcome.fits,
        "primary_probe_deg": outcome.primary_probe_deg,
        "frequencies_hz": outcome.frequencies_hz, "band": result.band,
        "calibration": result.calibration,
        "incidence_angle_rad": result.inversion.incidence_angle_rad,
        "result": result, "outcome": outcome})


# ---------------------------------------------------------------------------
# Save and load (inspection.npz)
# ---------------------------------------------------------------------------

def _flatten(prefix, instance, arrays):
    for item in fields(instance):
        value = getattr(instance, item.name)
        if value is None:
            continue
        if isinstance(value, dict):
            for name, entry in value.items():
                arrays[f"{prefix}{item.name}::{name}"] = np.asarray(entry)
        elif isinstance(value, list):
            continue
        else:
            arrays[prefix + item.name] = np.asarray(value)


def save_inspection(record, path):
    """Write the record as one ``.npz`` (series fields under ``series<i>/``)."""
    arrays = {}
    _flatten("", record, arrays)
    for position, entry in enumerate(record.series):
        _flatten(f"series{position}/", entry, arrays)
    np.savez_compressed(path, **arrays)
    return path


def _value(array, field_type):
    if array.dtype.kind in "US" and array.ndim == 0:
        return str(array)
    if array.ndim == 0:
        scalar = array.item()
        return int(scalar) if field_type is int or field_type == "int" else scalar
    return array


def _restore(cls, stored, prefix):
    values = {}
    for item in fields(cls):
        name = prefix + item.name
        if name in stored:
            values[item.name] = _value(stored[name], item.type)
        elif item.name not in ("series",):
            entries = {key[len(name) + 2:]: stored[key] for key in stored
                       if key.startswith(name + "::")}
            if entries:
                values[item.name] = entries
    return cls(**values)


def load_inspection(path):
    """Read an ``inspection.npz`` written by :func:`save_inspection` back into a record."""
    with np.load(path, allow_pickle=False) as handle:
        stored = {key: handle[key] for key in handle.files}
    count = len({key.split("/")[0] for key in stored if key.startswith("series")
                 and "/" in key})
    record = _restore(InspectionRecord, stored, "")
    record.series = [_restore(SeriesInspection, stored, f"series{position}/")
                     for position in range(count)]
    return record


# ---------------------------------------------------------------------------
# The figure registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InspectionFigure:
    name: str
    checkpoint: str
    function: object
    summary: str


#: name -> InspectionFigure. The one place an inspection figure exists.
INSPECTION_FIGURES: dict[str, InspectionFigure] = {}


def inspection_figure(checkpoint):
    """Register a figure of ``record`` drawn at ``checkpoint``; its docstring's first line is
    its description. The function takes an InspectionRecord and returns a Figure."""
    def register(function):
        summary = (function.__doc__ or "").strip().splitlines()
        INSPECTION_FIGURES[function.__name__] = InspectionFigure(
            name=function.__name__, checkpoint=checkpoint, function=function,
            summary=summary[0] if summary else "")
        return function
    return register


def _ensure_registered():
    from . import inspection_plots  # noqa: F401 -- registers the figures


def figures_at(checkpoint):
    _ensure_registered()
    return [entry for entry in INSPECTION_FIGURES.values() if entry.checkpoint == checkpoint]


def make_inspection_figures(record, checkpoints=None):
    """``{"<checkpoint>__<figure>.png": Figure}`` for the chosen checkpoints (default all)."""
    from .stages import CHECKPOINTS
    _ensure_registered()
    chosen = CHECKPOINTS if checkpoints is None else tuple(checkpoints)
    figures = {}
    for checkpoint in CHECKPOINTS:
        if checkpoint not in chosen:
            continue
        for entry in figures_at(checkpoint):
            figures[f"{checkpoint}__{entry.name}.png"] = entry.function(record)
    return figures


def save_figures(figures, directory, dpi=130):
    import matplotlib.pyplot as plt
    os.makedirs(directory, exist_ok=True)
    paths = []
    for filename, figure in figures.items():
        path = os.path.join(directory, filename)
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(path)
    return paths


def resolve_checkpoints(selection):
    """config['general']['inspect'] -> a tuple of checkpoint names (validated)."""
    from .stages import CHECKPOINTS
    if not selection:
        return ()
    if selection in ("all", True):
        return CHECKPOINTS
    chosen = (selection,) if isinstance(selection, str) else tuple(selection)
    unknown = [name for name in chosen if name not in CHECKPOINTS]
    if unknown:
        raise ValueError(f"unknown inspection checkpoint(s) {unknown}; known: {list(CHECKPOINTS)}")
    return chosen


class StepwiseInspector:
    """Observer for ``run_ellipsometry``: shows each chosen step's figures as the run reaches it.

    With ``block`` (default) each step's figures stay open until closed, and the run continues
    after -- the same rhythm as ``show_graph`` in the other pipelines. ``show=False`` builds the
    figures without displaying them (headless runs and tests); ``shown`` records what was drawn.
    """

    def __init__(self, checkpoints, *, show=True, block=True):
        self.checkpoints = resolve_checkpoints(checkpoints)
        self.show = show
        self.block = block
        self.shown = []

    def __call__(self, checkpoint, state):
        if checkpoint not in self.checkpoints:
            return
        import matplotlib.pyplot as plt
        figures = make_inspection_figures(build_inspection_record(state), (checkpoint,))
        self.shown.extend(figures)
        print(f"[inspect] {checkpoint}: {len(figures)} figure(s)"
              + (" -- close them to continue" if self.show and self.block else ""), flush=True)
        if self.show:
            plt.show(block=self.block)
        else:
            for figure in figures.values():
                plt.close(figure)
