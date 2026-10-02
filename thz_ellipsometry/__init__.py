"""THz time-domain ellipsometry.

A discrete, self-contained package for reflection ellipsometry on our bench. Deliberately
separate from ``dataset_core.adapters.thz_adapter`` so that its use stays clear and demarcated.

Layout and the one rule that matters:

    core/       PURE -- no DataSet, no file I/O, no global state except registries.
                Arrays in, arrays out. Imports only numpy, scipy and thz_core.
    adapters/   the only part that knows this repository and the bench exist: loading .acc
                files, writing synthetic ones, the stage driver, reports.

``core/`` is destined to become a component of the separate ``thz-core`` repository; a test
(``tests/test_thz_ellipsometry_core.py::test_core_imports_nothing_from_the_repository``)
enforces the boundary so the move stays a copy, not a refactor. This top-level namespace
re-exports the core API only -- import ``thz_ellipsometry.adapters`` explicitly when you want
the repo-aware half.

Phase 1 scope: isotropic samples, THz polarisation rotated at the emitter, GaP crystal fixed.
That is rank 2 of the Jones matrix -- complete for an isotropic sample such as silicon, and
rank-deficient for an anisotropic one. Phase 2 adds a second detection azimuth.

Design rationale, error budget and literature are in
``reports/thz_ellipsometry_prospecting_report.md``; the build spec is
``docs/ELLIPSOMETRY_MVP_PLAN.md`` (Phase 1 validation) and
``docs/THZ_ELLIPSOMETRY_IMPLEMENTATION_PLAN.md`` (the full build); findings are lab notebook
F33-F37.
"""

from __future__ import annotations

from .core.calibration import (
    ChannelCalibration,
    IncidenceAngleFit,
    channel_ratio_from_reference,
    fit_incidence_angle,
)
from .core.harmonic import HarmonicFit, fit_emitter_harmonic, harmonic_residual_norm
from .core.inversion import InversionResult, conductivity_from_index, invert_ratio
from .core.materials import (
    SILICON_HIGH_RESISTIVITY_INDEX,
    doped_silicon_index,
    gold_index,
    high_resistivity_silicon_index,
    reference_index,
)
from .core.calibration_sources import (
    CALIBRATION_SOURCES,
    CalibrationInputs,
    compute_channel_calibration,
)
from .core.detection import (
    BALANCED_PROBE_AZIMUTH_RAD,
    channel_ratio_for_probe,
    detection_vector_in_sample_frame,
    electro_optic_detection_vector,
    is_degenerate_azimuth,
)
from .core.model import ellipsometric_ratio, index_from_ellipsometric_ratio
from .core.pipeline import (
    EllipsometryResult,
    analyse_calibrated_series,
    analyse_polarisation_series,
    band_mask,
)
from .core.preprocess import TransformedSeries, spectra_from_traces, transform_traces
from .core.sensitivity import angle_error_affine_coefficients
from .core.tilt import DISPERSION_MODELS, TiltFit, fit_tilt_with_dispersion_model
from .core.simulate import (
    SyntheticMeasurement,
    interleaved_schedule,
    synthesize_acquisitions,
    synthesize_spectra,
)
from .core.validation import ValidationReport, validate_index_against_reference

__all__ = [
    "BALANCED_PROBE_AZIMUTH_RAD",
    "CALIBRATION_SOURCES",
    "CalibrationInputs",
    "ChannelCalibration",
    "DISPERSION_MODELS",
    "EllipsometryResult",
    "HarmonicFit",
    "IncidenceAngleFit",
    "InversionResult",
    "SILICON_HIGH_RESISTIVITY_INDEX",
    "SyntheticMeasurement",
    "TiltFit",
    "TransformedSeries",
    "ValidationReport",
    "analyse_calibrated_series",
    "analyse_polarisation_series",
    "angle_error_affine_coefficients",
    "band_mask",
    "channel_ratio_for_probe",
    "channel_ratio_from_reference",
    "compute_channel_calibration",
    "conductivity_from_index",
    "detection_vector_in_sample_frame",
    "doped_silicon_index",
    "electro_optic_detection_vector",
    "ellipsometric_ratio",
    "fit_emitter_harmonic",
    "fit_incidence_angle",
    "fit_tilt_with_dispersion_model",
    "gold_index",
    "harmonic_residual_norm",
    "high_resistivity_silicon_index",
    "index_from_ellipsometric_ratio",
    "interleaved_schedule",
    "invert_ratio",
    "is_degenerate_azimuth",
    "reference_index",
    "spectra_from_traces",
    "synthesize_acquisitions",
    "synthesize_spectra",
    "transform_traces",
    "validate_index_against_reference",
]
