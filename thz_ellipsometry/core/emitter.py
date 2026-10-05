"""The emitter: from a magnet reading to the THz polarisation angle it produces.

The spintronic emitter radiates perpendicular to its magnetisation, and the magnet is set by
hand against engraved marks, so the reading and the polarisation differ by an offset and,
possibly, by small per-state errors of the scale itself. Both matter: a polarisation error does
not scale rho, it MIXES the p and s channels (a Moebius map), and the gold calibration does not
absorb it.

Neither can be fitted from a sample measurement. With four magnet states and a background term
there is one complex number of redundancy per frequency, and per-state angle errors have no
distinct signature in it (singular-value ratio ~1e-8, on gold and on doped silicon alike). They
are therefore CALIBRATED, with the wire grid: grid passing s nulls the signal where the
polarisation is p or -p (0 and 180 deg), grid passing p where it is s or -s (90 and 270 deg).
Each null is found to a fraction of a degree by ``core.bench.fit_wire_grid_null``.

The calibration is a table {polarisation angle: magnet reading at that polarisation}. One entry
is a pure offset; several give a piecewise-linear, 360-deg-periodic map, so the same reading
always maps to the same polarisation -- which is what a reproducible manual setting needs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["polarization_from_reading", "reading_from_polarization", "validate_reading_table"]


def validate_reading_table(table):
    """Check a {polarisation_deg: reading_deg} table and return it as sorted arrays.

    Readings must increase with polarisation (one magnet sense) and span less than a full turn.
    """
    if not table:
        raise ValueError("the magnet calibration table is empty; give at least the reading "
                         "that emits p, e.g. {0.0: 86.3}")
    polarisations = np.array(sorted(float(key) for key in table))
    readings = np.array([float(table[key]) for key in sorted(table, key=float)])
    if polarisations.size > 1:
        unwrapped = readings[0] + np.mod(readings - readings[0], 360.0)
        steps = np.diff(unwrapped)
        if np.any(steps <= 0.0) or unwrapped[-1] - unwrapped[0] >= 360.0:
            raise ValueError(
                f"magnet readings {readings.tolist()} for polarisations "
                f"{polarisations.tolist()} do not increase together; check the rotation sense "
                "and the order of the nulls")
        readings = unwrapped
    return polarisations, readings


def polarization_from_reading(readings_deg, table):
    """THz polarisation angle (deg, from p) for each magnet reading, from a calibration table.

    One entry: polarisation = reading - offset. Several: linear interpolation between the
    calibrated states, periodic over 360 deg, so readings between calibrated states inherit the
    local scale and readings at them land exactly on them.
    """
    polarisations, readings = validate_reading_table(table)
    values = np.asarray(readings_deg, dtype=float)
    if polarisations.size == 1:
        return values - (readings[0] - polarisations[0])
    # Extend one period either side so interpolation is periodic.
    period_readings = np.concatenate([readings - 360.0, readings, readings + 360.0])
    period_polarisations = np.concatenate([polarisations - 360.0, polarisations,
                                           polarisations + 360.0])
    wrapped = readings[0] + np.mod(values - readings[0], 360.0)
    mapped = np.interp(wrapped, period_readings, period_polarisations)
    return mapped + (values - wrapped)


def reading_from_polarization(polarizations_deg, table):
    """Inverse of :func:`polarization_from_reading`: the reading that gives each polarisation.

    Used by the simulator to write the magnet READING a bench user would record.
    """
    polarisations, readings = validate_reading_table(table)
    values = np.asarray(polarizations_deg, dtype=float)
    if polarisations.size == 1:
        return values + (readings[0] - polarisations[0])
    period_polarisations = np.concatenate([polarisations - 360.0, polarisations,
                                           polarisations + 360.0])
    period_readings = np.concatenate([readings - 360.0, readings, readings + 360.0])
    wrapped = polarisations[0] + np.mod(values - polarisations[0], 360.0)
    mapped = np.interp(wrapped, period_polarisations, period_readings)
    return mapped + (values - wrapped)
