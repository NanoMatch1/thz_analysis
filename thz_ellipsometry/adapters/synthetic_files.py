"""Write synthetic measurements to disk in the bench's real ``.acc`` format.

This is the I/O half of the simulator: ``core.simulate`` makes the traces, this writes them so
the real loader reads them. The format follows real files exactly -- per-scan header blocks with
a ``Date and time`` parameter, scans separated by a ``%%`` line -- because a synthetic format
that only our own reader accepts would test nothing.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.simulate import synthesize_time_domain

__all__ = ["write_accumulation_files"]


def write_accumulation_files(directory, *, index_sample_function, emitter_angles_rad,
                             sample_name="sample", scans_per_angle=2,
                             angle_token="pol", **kwargs):
    """Write one ``.acc`` file per emitter angle, in the real on-disk format.

    Filenames follow the repo's grammar, ``<type>_<key>=<value>.acc``, e.g.
    ``silicon_pol=15.acc`` -- a key=value token that the existing parser extracts regardless of
    position. Each file holds ``scans_per_angle`` repeat scans, as a real accumulation does.

    Returns the list of paths written.
    """
    os.makedirs(directory, exist_ok=True)
    time_ps, traces = synthesize_time_domain(
        index_sample_function, emitter_angles_rad=emitter_angles_rad, **kwargs)

    written = []
    for position, angle in enumerate(np.asarray(emitter_angles_rad, dtype=float)):
        angle_deg = float(np.rad2deg(angle))
        label = f"{angle_deg:g}".replace("-", "m")
        path = os.path.join(directory, f"{sample_name}_{angle_token}={label}.acc")
        lines = []
        for scan_index in range(1, scans_per_angle + 1):
            if scan_index > 1:
                lines.append("%%")
            lines.append(f"%title {sample_name}_{angle_token}={label} acc {scan_index}")
            lines.append("%Created 3")
            lines.append("%type 0")
            lines.append("%Parameters Parameters")
            lines.append(
                f"%param Date and time,2026-10-01 "
                f"{10 + position // 60:02d}:{position % 60:02d}:{scan_index:02d}.000000")
            for time_value, amplitude in zip(time_ps, traces[position]):
                lines.append(f"{time_value:.18e} {amplitude:.18e}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        written.append(path)
    return written
