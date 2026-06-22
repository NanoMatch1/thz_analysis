"""Minimal standalone loader for .acc THz files.

Returns a dict::

    {
        "filename":         str,            # basename of the file
        "header":           list[str],      # header lines from the first scan
                                            #   (leading '%' stripped)
        "scan_headers":     list[list[str]] # per-scan header lines (one list
                                            #   per scan, '%' stripped)
        "data":             np.ndarray,     # float64, (N_points, 1+N_scans)
                                            #   col 0=time, cols 1…=signals
    }

.acc files contain multiple scans separated by ``%%``.  Each scan shares
the same time axis; only the signal column differs.  The returned ``data``
stacks them as ``[time | scan1 | scan2 | …]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_axis_gaps(data: np.ndarray, gap_threshold_factor: float = 1.5) -> np.ndarray:
    """Fill gaps in the time axis with zero-valued signal rows.

    Computes the nominal step from the median of all consecutive differences.
    Any step larger than ``nominal_step * gap_threshold_factor`` is treated as
    a gap and filled: new rows are inserted whose time advances by
    ``nominal_step`` and whose signal values are all 0.

    Parameters
    ----------
    data:
        Shape ``(N, 1 + S)`` — column 0 is time, columns 1..S are signals.
    gap_threshold_factor:
        Multiplier of the nominal step above which a jump is considered a gap.

    Returns
    -------
    np.ndarray
        Array with gaps filled.  Returned unchanged if no gaps are detected.
    """
    if data.shape[0] < 2:
        return data

    time = data[:, 0]
    diffs = np.diff(time)
    nominal_step = float(np.median(diffs))

    gap_indices = np.where(diffs > nominal_step * gap_threshold_factor)[0]
    if len(gap_indices) == 0:
        return data

    n_signal_cols = data.shape[1] - 1
    segments: list[np.ndarray] = []
    prev_idx = 0

    for gap_idx in gap_indices:
        segments.append(data[prev_idx : gap_idx + 1])

        t_before = time[gap_idx]
        t_after = time[gap_idx + 1]
        n_fill = round((t_after - t_before) / nominal_step) - 1

        if n_fill > 0:
            fill_time = t_before + nominal_step * np.arange(1, n_fill + 1)
            fill_signals = np.zeros((n_fill, n_signal_cols), dtype=np.float64)
            segments.append(np.column_stack([fill_time, fill_signals]))

        prev_idx = gap_idx + 1

    segments.append(data[prev_idx:])

    n_filled = sum(s.shape[0] for s in segments) - data.shape[0]
    print(
        f"  [gap fill] detected {len(gap_indices)} gap(s), "
        f"inserted {n_filled} zero-valued point(s)"
    )
    return np.vstack(segments)


def _parse_rows(rows: list[str]) -> np.ndarray:
    """Parse space-separated numeric rows into a 2-D float array."""
    numeric = []
    for row in rows:
        parts = row.split()
        if not parts:
            continue
        try:
            numeric.append([float(v) for v in parts])
        except ValueError:
            continue
    if not numeric:
        return np.empty((0, 0))
    return np.array(numeric, dtype=np.float64)


def _split_header_and_data(text: str):
    """Return (header_lines, data_rows) from a block of text."""
    header: list[str] = []
    data: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("%"):
            header.append(line.lstrip("%").strip())
        else:
            data.append(line)
    return header, data


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".acc",)


def load_acc(filepath: Union[str, Path]) -> dict:
    """Load an ``.acc`` file and return a standardised dict.

    Multiple scans (delimited by ``%%``) are combined so that ``data`` has
    shape ``(N_points, 1 + N_scans)``.  Per-scan headers are preserved in
    ``scan_headers``.
    """
    filepath = Path(filepath)
    raw_text = filepath.read_text(encoding="utf-8")

    blocks = raw_text.split("%%")

    scan_headers: list[list[str]] = []
    signal_columns: list[np.ndarray] = []
    time_axis: np.ndarray | None = None

    for block in blocks:
        header, rows = _split_header_and_data(block)
        if not rows:
            continue
        arr = _parse_rows(rows)
        if arr.size == 0:
            continue

        scan_headers.append(header)

        if time_axis is None:
            time_axis = arr[:, 0]

        if arr.shape[1] >= 2:
            for col_idx in range(1, arr.shape[1]):
                signal_columns.append(arr[:, col_idx])

    if time_axis is None:
        return {
            "filename": filepath.name,
            "header": [],
            "scan_headers": [],
            "data": np.empty((0, 0)),
        }

    data = np.column_stack([time_axis] + signal_columns)
    data = _fill_axis_gaps(data)

    return {
        "filename": filepath.name,
        "header": scan_headers[0] if scan_headers else [],
        "scan_headers": scan_headers,
        "data": data,
    }


def load_file(filepath: Union[str, Path]) -> dict:
    """Load a ``.acc`` file.

    Raises
    ------
    ValueError
        If the extension is not ``.acc``.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext != ".acc":
        raise ValueError(
            f"Unsupported extension '{ext}'. Only .acc files are supported."
        )
    return load_acc(filepath)
