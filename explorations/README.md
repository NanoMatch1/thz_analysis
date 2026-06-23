# explorations/

Throwaway / scoping scripts that read thz_core + dataset_core but are **not** part of the
pipeline. Grouped by topic. Run from the repo root with the repo on the path, e.g.
`PYTHONPATH=. .venv/Scripts/python.exe explorations/<group>/<script>.py`.

- **`air_gap_cnt_reflection/`** — air-gap de-embedding for the CNT window-reflection samples
  (the n<1 problem). Has its own **`EXPLORATION_LOG.md`** with the full problem→hypothesis→
  conclusion record. Active line of work.
- **`sandwich_transmission_cuvette/`** — air|sub|sample|sub|air transmission: MATLAB/Novelli
  cross-validation and the fused-silica cuvette preprocessing study.
- **`reflection_phase_and_windowing/`** — reflection phase audit, phase-unwrap vs legacy, and
  windowing/centering demos that fed the reflection pipeline design.
- **`selfreferencing_validation/`** — window self-referencing evaluation and consistency checks.
- **`test_runners/`** — convenience runners for the thz_core test suite.
- `output/` — scratch outputs.
