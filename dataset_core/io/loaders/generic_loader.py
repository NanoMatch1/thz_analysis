"""Generic fallback loader for text files with unknown or unsupported formatting.

Used when no registered loader matches the file extension, or when a registered
loader raises an error on a known extension with unexpected formatting.

The loader:
- Sniffs the delimiter and comment/header lines automatically.
- Extracts the first two numeric columns found.
- Returns a ``Spectrum`` object.

To convert the result to ``THzData`` for use in the THz pipeline, call
``spectrum_to_thzdata(spectrum)``.
"""

from __future__ import annotations

import csv
import re
import warnings
from os import path
from typing import List, Optional, Tuple

import numpy as np

from dataset_core.data_structures.spectrum import Spectrum
from dataset_core.data_structures.thz import THzData, BaseTHzData


# ---------------------------------------------------------------------------
# Delimiter sniffing helpers
# ---------------------------------------------------------------------------

def _sniff_delimiter(sample_lines: List[str]) -> str:
    """Detect the column delimiter from a sample of lines.

    Tries ``csv.Sniffer`` on data-looking lines first, then falls back to
    counting occurrences of common delimiters.
    """
    data_sample = "\n".join(
        line for line in sample_lines
        if line.strip() and _looks_numeric(line.split()[0] if line.split() else "")
    )[:2000]

    if data_sample:
        try:
            dialect = csv.Sniffer().sniff(data_sample)
            return dialect.delimiter
        except csv.Error:
            pass

    # Heuristic: pick the delimiter most consistently present in data lines
    candidates = [",", "\t", ";", " "]
    full_sample = "\n".join(sample_lines[:50])
    counts = {d: full_sample.count(d) for d in candidates}
    return max(counts, key=counts.get)


def _looks_numeric(token: str) -> bool:
    """Return True if *token* looks like a float/int, including scientific notation."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def _detect_skip_rows(lines: List[str], delimiter: str) -> int:
    """Return the index of the first line that parses entirely as numeric columns.

    This is used to count how many header rows to skip.
    """
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if delimiter == " ":
            tokens = stripped.split()
        else:
            tokens = [t.strip() for t in stripped.split(delimiter)]
        tokens = [t for t in tokens if t]
        if tokens and all(_looks_numeric(t) for t in tokens):
            return idx
    return 0


# ---------------------------------------------------------------------------
# Column extraction
# ---------------------------------------------------------------------------

def _extract_two_columns(
    lines: List[str], delimiter: str, skip_rows: int
) -> Tuple[np.ndarray, List[str]]:
    """Parse *lines* and extract the first two numeric columns.

    Lines before *skip_rows* are treated as headers.  Lines that cannot be
    parsed as all-numeric are also collected as header/comment strings.

    Returns
    -------
    data : np.ndarray, shape (N, 2)
        First two numeric columns; NaN where a row had fewer than two values.
    headers : list of str
        Non-numeric lines encountered anywhere in the file.
    """
    headers: List[str] = list(lines[:skip_rows])
    numeric_rows: List[List[float]] = []

    for line in lines[skip_rows:]:
        stripped = line.strip()
        if not stripped:
            continue

        if delimiter == " ":
            tokens = stripped.split()
        else:
            reader = csv.reader([stripped], delimiter=delimiter)
            tokens = next(reader)
        tokens = [t.strip() for t in tokens if t.strip()]

        try:
            row = [float(t) for t in tokens]
        except ValueError:
            headers.append(line)
            continue

        if len(row) == 0:
            continue

        numeric_rows.append(row)

    if not numeric_rows:
        return np.empty((0, 2), dtype=float), headers

    # Identify two best columns: prefer first two, but skip constant columns
    # when there are more than two so we don't grab an index column of all zeros
    max_cols = max(len(r) for r in numeric_rows)
    # Pad to rectangular
    arr = np.full((len(numeric_rows), max_cols), np.nan, dtype=float)
    for i, row in enumerate(numeric_rows):
        arr[i, : len(row)] = row

    if max_cols >= 2:
        data = arr[:, :2]
    else:
        # Only one column — synthesize an integer X axis
        x = np.arange(arr.shape[0], dtype=float)
        data = np.column_stack((x, arr[:, 0]))

    return data, headers


# ---------------------------------------------------------------------------
# Public loader class
# ---------------------------------------------------------------------------

class GenericTextLoader:
    """Fallback loader for text files whose format is not handled by a registered loader.

    Unlike registered loaders, this class is not bound to a specific file
    extension.  Instantiate it with any text file path and call ``load()``.

    Returns a ``Spectrum`` object.  Use ``spectrum_to_thzdata`` to convert the
    result to ``THzData`` if needed.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = path.basename(filepath)

    def load(self) -> Spectrum:
        """Read the file, sniff formatting, extract two numeric columns.

        Raises
        ------
        ValueError
            If the file contains no parseable numeric data.
        """
        try:
            with open(self.filepath, "r", encoding="utf-8", errors="replace") as fh:
                raw_text = fh.read()
        except OSError as exc:
            raise ValueError(f"Cannot read '{self.filepath}': {exc}") from exc

        lines = raw_text.splitlines()
        if not lines:
            raise ValueError(f"File '{self.filepath}' is empty.")

        sample = lines[:50]
        delimiter = _sniff_delimiter(sample)
        skip_rows = _detect_skip_rows(lines, delimiter)

        data, headers = _extract_two_columns(lines, delimiter, skip_rows)

        if data.size == 0:
            raise ValueError(
                f"No numeric data found in '{self.filepath}'. "
                "The file may be binary or have an unsupported format."
            )

        warnings.warn(
            f"GenericTextLoader used for '{self.filename}'. "
            f"Detected delimiter={delimiter!r}, skipped {skip_rows} header row(s). "
            "Check that the loaded data looks correct.",
            UserWarning,
            stacklevel=2,
        )

        return Spectrum(
            data=data,
            headers=headers,
            filename=self.filename,
            data_type="generic",
        )


# ---------------------------------------------------------------------------
# Conversion helper: Spectrum → THzData
# ---------------------------------------------------------------------------

def spectrum_to_thzdata(
    spectrum: Spectrum,
    *,
    time_unit_scale: float = 1.0,
    data_type: str = "generic",
) -> THzData:
    """Convert a ``Spectrum`` to a ``THzData`` object for use in the THz pipeline.

    The Spectrum's ``x`` column is used as the time axis and ``y`` as the
    amplitude.  The returned ``THzData`` contains a single synthetic scan.

    Parameters
    ----------
    spectrum : Spectrum
        Source data, typically produced by ``GenericTextLoader.load()``.
    time_unit_scale : float
        Multiply the x-axis by this factor before storing.  Defaults to 1.0
        (no conversion).  Pass ``1e-12`` if the source x-axis is in picoseconds
        and you want the ``BaseTHzData.raw_data`` stored in seconds so that
        ``THzData._average_data`` does *not* apply an additional ``1e-12``
        scale.  Leave as 1.0 if the x-axis is already in picoseconds (the
        default THz pipeline convention for raw data).
    data_type : str
        Label stored on the resulting ``THzData``. Defaults to ``'generic'``.

    Returns
    -------
    THzData
    """
    if spectrum.data.shape[1] < 2:
        raise ValueError("Spectrum must have at least 2 columns (x, y) to convert to THzData.")

    x = spectrum.x * time_unit_scale
    y = spectrum.y
    scan_array = np.column_stack((x, y))

    single_scan = BaseTHzData(
        data=scan_array,
        headers=spectrum.headers if spectrum.headers is not None else [],
    )

    return THzData(
        data=[single_scan],
        header=spectrum.headers,
        filename=spectrum.filename or "unknown_file",
        data_type=data_type,
    )
