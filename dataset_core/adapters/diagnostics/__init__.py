"""Runtime assumption checking for the THz pipeline.

    from dataset_core.adapters import diagnostics

    diagnostics.check(dataset)            # run the checks, record them
    diagnostics.run_report(dataset)       # end-of-run health report
    diagnostics.write_ledger("docs/assumptions_ledger.md")

Diagnostics observe and report; they never halt a run. Importing this package
registers the built-in checks.
"""

from __future__ import annotations

from .registry import (
    DIAGNOSTIC_REGISTRY,
    Diagnostic,
    Finding,
    Problem,
    Severity,
    diagnostic,
    registered_stages,
    render_assumptions_ledger,
    run_diagnostics,
)
from .run_log import Note, RunLog, check, get_run_log, run_report

from . import checks as _builtin_checks  # noqa: F401 — registers the built-ins

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "Diagnostic",
    "Finding",
    "Note",
    "Problem",
    "RunLog",
    "Severity",
    "check",
    "diagnostic",
    "get_run_log",
    "registered_stages",
    "render_assumptions_ledger",
    "run_diagnostics",
    "run_report",
    "write_ledger",
]


def write_ledger(path: str) -> str:
    """Generate the assumptions ledger from the registry and write it to ``path``."""
    text = render_assumptions_ledger()
    with open(path, "w") as handle:
        handle.write(text)
    print(f"[diagnostics] wrote {len(DIAGNOSTIC_REGISTRY)} registered assumptions "
          f"to {path}")
    return text
