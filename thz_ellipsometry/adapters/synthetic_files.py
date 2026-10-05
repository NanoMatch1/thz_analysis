"""Write synthetic measurements to disk in the bench's real ``.acc`` format.

This is the I/O half of the simulator: ``core.simulate`` makes the scans, this writes them so
the real loader reads them. The format follows real files exactly -- per-scan header blocks with
a ``Date and time`` parameter, scans separated by a ``%%`` line -- because a synthetic format
that only our own reader accepts would test nothing. The timestamps are the schedule's real
elapsed times, so an interleaved acquisition looks on disk exactly as it will on the bench.
"""

from __future__ import annotations

import datetime
import os

import numpy as np

from ..core.emitter import reading_from_polarization
from ..core.simulate import interleaved_schedule, synthesize_acquisitions

__all__ = ["write_accumulation_files"]

DEFAULT_START = datetime.datetime(2026, 10, 1, 10, 0, 0)


def _angle_label(angle_deg):
    return f"{angle_deg:g}".replace("-", "m")


def write_accumulation_files(directory, *, index_sample_function, emitter_angles_rad=None,
                             schedule=None, sample_name="sample", scans_per_angle=2,
                             angle_token="pol", magnet_calibration=None, probe_token=None,
                             probe_azimuth_deg=None, start_time=DEFAULT_START,
                             time_offset_seconds=0.0, **kwargs):
    """Write one ``.acc`` file per acquisition, in the real on-disk format.

    Give either ``emitter_angles_rad`` (one acquisition per angle, in order) or a ``schedule``
    from :func:`~thz_ellipsometry.core.simulate.interleaved_schedule`. Angles are THz
    polarisation from p; the filename records the MAGNET reading that gives it under the TRUE
    ``magnet_calibration`` table ({polarisation: reading}), as the bench will. With ``probe_token`` set, the probe setting is
    written into the name too. Cycles beyond the first add a ``cyc=NN`` token so the names stay
    unique. ``time_offset_seconds`` shifts this object's whole schedule, for writing a sample and
    its reference as consecutive blocks.

    ``probe_azimuth_deg`` is passed on to the simulator as the true probe setting. Remaining
    keyword arguments go to :func:`~thz_ellipsometry.core.simulate.synthesize_acquisitions`.

    Returns the list of paths written.
    """
    if (emitter_angles_rad is None) == (schedule is None):
        raise ValueError("give exactly one of emitter_angles_rad or schedule")
    seconds_per_scan = kwargs.pop("seconds_per_scan", 10.0)
    if schedule is None:
        schedule = interleaved_schedule(np.rad2deg(emitter_angles_rad), cycles=1,
                                        seconds_per_acquisition=scans_per_angle
                                        * seconds_per_scan)
    if probe_azimuth_deg is not None:
        kwargs["probe_azimuth_rad"] = np.deg2rad(probe_azimuth_deg)

    os.makedirs(directory, exist_ok=True)
    elapsed = schedule.elapsed_seconds + float(time_offset_seconds)
    acquisitions = synthesize_acquisitions(
        index_sample_function, polarization_angles_rad=schedule.polarization_angles_rad,
        elapsed_seconds=elapsed, scans_per_acquisition=scans_per_angle,
        seconds_per_scan=seconds_per_scan, **kwargs)

    several_cycles = int(np.max(schedule.cycle_numbers)) > 1
    written = []
    for position, angle in enumerate(schedule.polarization_angles_rad):
        magnet_deg = round(float(reading_from_polarization(np.rad2deg(angle),
                                                           magnet_calibration or {0.0: 0.0})), 3)
        stem = f"{sample_name}_{angle_token}={_angle_label(magnet_deg)}"
        if probe_token is not None and probe_azimuth_deg is not None:
            stem += f"_{probe_token}={_angle_label(probe_azimuth_deg)}"
        if several_cycles:
            stem += f"_cyc={int(schedule.cycle_numbers[position]):02d}"
        lines = []
        for scan_index in range(scans_per_angle):
            if scan_index:
                lines.append("%%")
            stamp = start_time + datetime.timedelta(
                seconds=float(elapsed[position]) + scan_index * seconds_per_scan)
            lines.append(f"%title {stem} acc {scan_index + 1}")
            lines.append("%Created 3")
            lines.append("%type 0")
            lines.append("%Parameters Parameters")
            lines.append(f"%param Date and time,{stamp.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            for time_value, amplitude in zip(acquisitions.time_ps,
                                             acquisitions.scans[position, scan_index]):
                lines.append(f"{time_value:.18e} {amplitude:.18e}")
        path = os.path.join(directory, f"{stem}.acc")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        written.append(path)
    return written
