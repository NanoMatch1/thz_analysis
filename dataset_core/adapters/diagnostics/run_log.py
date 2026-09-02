"""The run log — what happened during a run, collected and reported at the end.

Individual stages already print a ``[stage_name] ...`` line as they go, which is good
for watching a run but useless afterwards: the interesting lines scroll past, and by
the time a result looks odd the evidence is 300 lines up. The run log keeps them.

It holds two kinds of entry:

- **notes**, added by stages themselves — what a stage did and to what;
- **findings**, produced by the diagnostics registry — assumptions that stopped
  holding.

The log lives on the dataset, so it travels with a saved session and a reopened
analysis can still answer "was this measurement healthy?" without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .registry import Finding, Severity, run_diagnostics

_RUN_LOG_ATTRIBUTE = "_run_log"


@dataclass
class Note:
    """Something a stage did, worth keeping in the record."""

    stage: str
    message: str
    filename: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class RunLog:
    """Everything worth knowing about a run, in order."""

    notes: list[Note] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def note(self, stage: str, message: str, *, filename: str | None = None,
             **detail) -> None:
        self.notes.append(Note(stage=stage, message=message, filename=filename,
                               detail=detail))

    def extend(self, findings) -> None:
        self.findings.extend(findings)

    def counts(self) -> dict:
        tally = {severity: 0 for severity in Severity}
        for finding in self.findings:
            tally[finding.severity] += 1
        return tally

    @property
    def worst(self) -> Severity | None:
        return max((finding.severity for finding in self.findings), default=None)

    def to_dict(self) -> dict:
        """Plain data, for storing alongside a saved session."""
        return {
            "notes": [asdict(note) for note in self.notes],
            "findings": [
                {**asdict(finding), "severity": finding.severity.label}
                for finding in self.findings
            ],
        }


def get_run_log(dataset) -> RunLog:
    """The dataset's run log, created on first use."""
    log = getattr(dataset, _RUN_LOG_ATTRIBUTE, None)
    if log is None:
        log = RunLog()
        setattr(dataset, _RUN_LOG_ATTRIBUTE, log)
    return log


def check(dataset, config: dict | None = None, *, stage: str | None = None) -> list:
    """Run the diagnostics and record the findings in the dataset's run log."""
    findings = run_diagnostics(dataset, config, stage=stage)
    get_run_log(dataset).extend(findings)
    return findings


def run_report(dataset, config: dict | None = None, *, rerun: bool = True,
               show_notes: bool = False) -> str:
    """Build the end-of-run report. Returns it as text and prints it.

    ``rerun`` re-runs every diagnostic against the dataset's final state, which is
    usually what you want at the end of a script — an assumption can be fine when a
    stage runs and strained by the time a later stage has reshaped the data.
    """
    log = get_run_log(dataset)
    if rerun:
        log.findings = []
        log.extend(run_diagnostics(dataset, config))

    tally = log.counts()
    lines = ["", "=" * 78, "RUN REPORT", "=" * 78]

    if show_notes and log.notes:
        lines.append("")
        lines.append("What ran:")
        for note in log.notes:
            target = f" [{note.filename}]" if note.filename else ""
            lines.append(f"    {note.stage}{target}: {note.message}")

    if not log.findings:
        lines += ["", "  No assumptions were found broken. That is not a guarantee of a",
                  "  good measurement — only that everything currently checked held.", ""]
        report = "\n".join(lines)
        print(report)
        return report

    lines.append("")
    lines.append(f"  {tally[Severity.FAIL]} broken, {tally[Severity.WARN]} strained, "
                 f"{tally[Severity.INFO]} noted. Nothing was halted.")

    for severity in (Severity.FAIL, Severity.WARN, Severity.INFO):
        matching = [finding for finding in log.findings if finding.severity == severity]
        if not matching:
            continue
        heading = {Severity.FAIL: "BROKEN — results depending on these are not usable",
                   Severity.WARN: "STRAINED — usable, but qualified",
                   Severity.INFO: "NOTED"}[severity]
        lines += ["", f"  {heading}", "  " + "-" * (len(heading))]
        for finding in matching:
            lines.append(finding.format(indent="  "))

    lines.append("")
    report = "\n".join(lines)
    print(report)
    return report
