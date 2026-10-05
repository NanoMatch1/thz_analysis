"""Notes for a saved analysis, asked for at the END of a run, when the result is on screen.

Why at the end: notes entered in a config before processing are skipped or stale -- the same
script is re-pointed at new directories and the old note rides along -- which is why the
catalogue is hard to read back. At save time the person knows what the result shows.

What this does:

- **Pre-fills** what the run already knows (script, data, samples, flags), so the person only
  writes the meaning: what the dataset is and why it was run.
- **Never blocks automation.** It only prompts on an interactive terminal. Headless runs, tests
  and piped input fall back to the config's notes (or none). ``THZ_NOTES_MODE`` overrides the
  mode for a whole session (e.g. ``THZ_NOTES_MODE=none`` on a batch machine).
- Records **where the note came from** (prompt, config, none), so provenance says whether a
  human described the run.

Notes can be corrected later with ``catalog_browse.py --annotate`` -- the prompt is the convenient
path, not the only one.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

__all__ = ["NOTES_MODES", "RunNotes", "collect_run_notes"]

#: mode -> one-line meaning. 'auto' prompts only when a person is at an interactive terminal.
NOTES_MODES = {
    "auto": "prompt on an interactive terminal, otherwise use the config notes",
    "prompt": "always prompt (still falls back if there is no input stream)",
    "config": "use the notes given in the config, never prompt",
    "none": "save without notes",
}
ENVIRONMENT_VARIABLE = "THZ_NOTES_MODE"
SKIP_TOKEN = "-"


@dataclass(frozen=True)
class RunNotes:
    text: str                 #: what the person wrote (or the config notes)
    source: str               #: 'prompt' | 'config' | 'none'
    prefilled_summary: str    #: what the run knew without being told

    def as_record(self):
        return {"text": self.text, "source": self.source,
                "prefilled_summary": self.prefilled_summary}


def _interactive(stream):
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _prompt(prefilled_summary, config_notes, input_function, output_function):
    output_function("")
    output_function("=" * 78)
    output_function("SAVE: describe this dataset for the catalogue")
    output_function("=" * 78)
    output_function(prefilled_summary)
    output_function("-" * 78)
    output_function("What is this dataset and why was it run? Several lines are fine; finish with "
                    "an empty line.")
    if config_notes:
        output_function(f"Press Enter at once to keep the config notes: {config_notes!r}")
    output_function(f"Type '{SKIP_TOKEN}' to save without notes.")
    lines = []
    while True:
        try:
            line = input_function("> " if not lines else "  ")
        except EOFError:
            break
        if not lines and line.strip() == SKIP_TOKEN:
            return "", "none"
        if not line.strip():
            break
        lines.append(line.rstrip())
    if lines:
        return "\n".join(lines), "prompt"
    if config_notes:
        return config_notes, "config"
    return "", "none"


def collect_run_notes(prefilled_summary, *, mode="auto", config_notes="",
                      input_function=input, output_function=print, input_stream=None,
                      environment=None):
    """Return the notes to save with a run. Never raises for lack of a terminal.

    Parameters
    ----------
    prefilled_summary : str      what the run knows, shown above the prompt and always recorded
    mode : str                   one of NOTES_MODES; ``THZ_NOTES_MODE`` in the environment wins
    config_notes : str           notes from the config, the fallback (and the 'config' answer)
    input_function, output_function, input_stream, environment
                                 injectable for tests; default to the real terminal
    """
    environment = os.environ if environment is None else environment
    mode = environment.get(ENVIRONMENT_VARIABLE, mode) or "auto"
    if mode not in NOTES_MODES:
        raise ValueError(f"unknown notes mode {mode!r}; known: {sorted(NOTES_MODES)} "
                         f"(set by config or {ENVIRONMENT_VARIABLE})")
    config_notes = (config_notes or "").strip()
    stream = sys.stdin if input_stream is None else input_stream

    if mode == "none":
        return RunNotes("", "none", prefilled_summary)
    if mode == "config" or (mode == "auto" and not _interactive(stream)):
        return RunNotes(config_notes, "config" if config_notes else "none", prefilled_summary)
    text, source = _prompt(prefilled_summary, config_notes, input_function, output_function)
    return RunNotes(text, source, prefilled_summary)
