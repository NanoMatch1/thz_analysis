# Lab web apps and packaged tools — proposal and evaluation

*2026-10-05. Samuel's ideas (brain dump after the ellipsometry bench prep), evaluated. Spans three
repos: `TDS-app` (acquisition), `thz-core` (science), `thz_analysis` (pipeline). Status of this
proposal lives in `~/.claude/global_projects.md`, not here.*

**Parked until the ellipsometry work is finished. This is the starting point to come back to.**
It lives in `thz_analysis` because the tools, the catalogue and the pipeline are here; the
TDS-app memory points at it. When the build starts, the web app's code belongs in TDS-app (it is
the bench's UI), the tool contract in thz-core (shared by both apps), and the bundle browser here.

## Decisions taken (2026-10-05, Samuel's replies)

- **Diagnostics only, for now.** The first web app is the read-only health page. Hardware control
  (motors, split scan, setup) is deferred; leave room for it, build none of it.
- **The tool contract is agreed as described in 1.2:** the app knows nothing about any individual
  tool; everything (help, tooltips, parameters, payload) comes with the tool.
- **Humidity: warn, never block; correct, don't wait.** Samuel's operating numbers: at least
  **30 min after a fully opened box**, at least **10 min after a small lid lift** for an optic
  adjustment. The criterion that matters is "did the spectrum change from reference to sample", so
  a little residual water is fine as long as it is CONSISTENT across what is being ratioed. (He is
  also considering a glove-port lid, which removes the lid-lift disturbance.)
- **Bundle notes move to the END of the script**, as a prompt on saving (§1.5a), because entering
  them up front is the reason the catalogue is hard to read.
- **Two separate apps** confirmed.
- **Health page runs locally** on the acquisition PC; the out-of-memory machine is getting a
  factory reset, expected to be enough.
- **Data on disk:** the py2 app writes the `.acc` at the end (save button), and probably a backup
  after each acquisition — to be confirmed. If confirmed, the folder-watching route gets
  per-acquisition cadence; if not, the health page needs that backup or an event stream.
- **`tds_core` phase 1 is committed** (Samuel, 2026-10-05).
- **The browser indexes the catalogue root only.** `~/data/data_sync` is a transfer route to the Pi
  for development, not a data home.

---

## 0. Two facts that change the framing

1. **The Python 3 direction is already decided, and partly built.** On 2026-09-10 Samuel decided
   (TDS-app memory `py3_port_decisions.md`): headless `tds_core`, no Qt port, a **web UI replaces
   the desktop GUI and eventually absorbs `orchestrator/experiment_gui.py`**, keep serving the
   9100/9101 socket protocol during the overlap, **reuse `thz-core/metrics.py` for live quality
   verdicts**, bind to localhost + VPN, single-writer control lock. Phase 1 of `tds_core` is on
   `feature/tds-core` (committed by Samuel 2026-10-05).
   So the question is not "py2 or py3 for the web app" — the web app is py3 by decision. The
   question is only how long the py2 app stays the acquisition engine underneath it.

2. **The tools already have three registries that each work the same way:** `@diagnostic`
   (dataset_core: check + assumption/why/remedy → runtime check and generated ledger),
   `@bench_tool` (thz_ellipsometry: CLI subcommands + help), `@ellipsometry_stage`. The packaging
   Samuel describes is these converging on ONE tool contract, not a new system.

---

## 1. The ideas, evaluated

### 1.1 One web app coordinating the bench (motors via REPL, split scan, setup, T0/slab, diagnostics) — **YES, staged; control comes last**

Valuable: the pieces exist separately (motor setup GUI, setup profiles, split scan, REPL, socket
server, window-mode T0/slab), and the comprehension debt Samuel names is real.

Push back on the ORDER, not the idea. A control surface is the expensive, risky part: it needs
the single-writer lock, safe stop, and parity with what the py2 app does. A read-only
**health page** gets all the diagnostic value with none of that risk, proves the architecture
(tool contract, streaming, reports), and is useful on the next bench day. Control features then
land into a proven shell. See the staging in §3.

### 1.2 Tools packaged with help, tooltips, hover icons — **YES; make the tool contract the single source of the UI**

To actually avoid comprehension debt, the help must live with the code and the UI must be
GENERATED from it, exactly as `@diagnostic` generates the assumptions ledger. A tool declares:

| Field | Becomes |
|---|---|
| `name`, `summary` (one line) | menu entry, tooltip |
| `help` (markdown: what it measures, why it matters, how to read it) | the hover/"?" panel and the generated docs page |
| `parameters`: name, unit, default, range, one-line description | the settings form, each field's tooltip |
| `inputs`: what data it needs (scans of one state, a full block, a reference...) | when the UI can enable it |
| `outputs`: the `thz-core` metrics payload — `values`, `flags`, `warnings` | the panel, the report |
| `default_enabled`, `cadence` (per scan / per file / per block) | the health-page defaults |
| `advisory_levels`: what is info / warn / act | the colour of the flag |

Rules that keep it clean: tools are **pure** (arrays + config in, payload out; no web, no file
I/O — the `core/` rule we already enforce with an import test), **deterministic**, and each has a
known-answer test. The web app knows nothing about any specific tool: it renders specs. A new
tool is one decorated function plus its test, and it appears in the UI, the docs and the report.

Polish: one help page per tool, generated, also printed by the CLI (`... help <tool>`) — the
bench CLI and the web page then can never disagree.

### 1.3 Live health page: streaming quality, drift, abnormal scans, advisory flags — **YES; this is the highest-value piece**

Directly serves the "active diagnostics" principle (no 6-hour run with a blocked beam). Candidate
tools, all of which exist or are in flight:

| Tool | Source | Cadence |
|---|---|---|
| Purge settling (delay / amplitude rate, fs/min) | `bench live` (built) | per scan |
| Abnormal scan (outlier vs. the file's other scans) | `thz_core.noise.drift_corrected_scatter` residual per scan | per scan |
| High-frequency SNR at the band edge ("is the purge good enough to use?") | repeat-scan noise model | per file |
| Purge spectral tilt / water template | existing `purge_still_equilibrating`; water matched filter (planned) | per file |
| Harmonic-model residual, drift, amplitude, background | `thz_ellipsometry` fit | per block |
| Magnet null / calibration status | `bench null` (built) | on demand |
| HWP walk | `bench hwp` (built) | on demand |
| T0 / slab in back-reflection window mode | existing reflection code (to be wrapped) | per file |

**Humidity: agree completely — warn, never block, and make the advisory say whether the drift is
CORRECTABLE, not whether it exists.** Waiting 2–4 h for a static purge is not the operating mode
when the box opens regularly. The useful verdict is two numbers: (a) usable high-frequency SNR at
the top of the band, and (b) whether the drift model leaves residuals at the noise level
(chi-square ~1). "Drifting but correctable" is green-amber, not red. One caveat to keep honest:
the fast transient in the first minutes after closing is NOT what a linear-ramp model corrects,
so a short wait (minutes, measured from the August purge data) still applies — the tool should
report that wait, not the full equilibration time.

Push back on one framing: "purge level" as an absolute number needs a calibrated hygrometer or a
water-line fit against a reference; the THz data alone gives the *change* in humidity, robustly.
Advise on change and usability, not on absolute RH, unless a sensor is added.

### 1.4 Diagnostic report dumped next to the data, importable by the pipeline — **YES, with two design rules**

- **Machine-readable first:** `diagnostics.ndjson` (one record per tool evaluation, appended as
  the run goes, so a crash loses nothing) plus a `diagnostics_summary.json` at the end; a short
  `diagnostics.md` for humans. Every record carries tool name + version, git SHA, config, and the
  file(s) it judged. Schema-versioned.
- **The live record is a log of what the operator saw, not the source of truth.** The pipeline
  re-runs the same (deterministic, versioned) tools on the files and compares. If the live and
  offline verdicts differ, that is itself a finding (different config, or the data changed). This
  avoids the classic split where the live app and the pipeline compute "the same" metric two ways.
- The `.gitignore` rule from the TDS-app cleanup (`**/data/**/*.dat|acc|ndjson`) already covers
  sidecars in data folders.

### 1.5a Bundle notes at the end, as a save prompt — **YES, with three refinements**

Samuel's diagnosis: notes go in the config before processing, often for a directory change on the
same script, so they are skipped or wrong. Asking at the end, when the result is on screen, gets
better descriptions.

- **Pre-fill what the machine knows** (script, input directory, sample/reference names, key config
  values, diagnostic flags, git SHA), so the prompt asks only for the meaning: "what is this
  dataset and why was it run".
- **Never block automation.** Headless runs and tests must not hang on `input()`: a config flag or
  environment variable selects "prompt", "use config notes" or "none", and a non-interactive
  session falls back to the pre-filled summary.
- **Make notes editable afterwards** (`catalog_browse.py --annotate <bundle_id>`, re-indexed), so a
  run saved in a hurry can be described properly later — the prompt is the convenient path, not
  the only one.

### 1.5 A second web app to browse processed bundles — **YES, cheap, but fix provenance first**

The index already exists: `dataset_core/adapters/catalog` + `catalog_browse.py` (find by type,
sample, date, fits, flags; open in the results viewer). A web page over it is a thin UI. But the
"which run_me made this?" problem is mostly **provenance**, and the catalog can only answer what
the bundles record:

- Only **3 of 12** run_me scripts write bundles (`low-level`, `reflection_single`, `transmission`).
  Ellipsometry, dual-pol, acquisition drift, fitting and the rest leave nothing indexable.
- Bundles record the git SHA and config, but **not the producing script** or its config-dict name.

So: (1) every run_me writes a bundle with `producer` (script path + name), git SHA, config, input
directory, and the diagnostics sidecar if one exists; (2) then the web viewer. Its key feature is
not plots but **"copy a load snippet"** — the three lines of Python that reload exactly this
bundle — which is the bridge back to custom figures in code that Samuel described.

**Keep the two apps separate.** They run on different machines with different jobs: the
acquisition app on the acquisition PC (single-writer, localhost + VPN), the data browser on the
analysis machine. They share the tool contract and the plotting component, not a server.

---

## 2. Practical constraints to check before building

- **Memory on the acquisition PC.** It ran out of memory with the py2 app resident (OpenBLAS
  failure, 2026-08-26). Decision: run locally after a factory reset. Still pin
  `OPENBLAS_NUM_THREADS=1` in the backend, and measure.
- **Where the data stream comes from.** Two options: (a) watch the data folder — works with the
  py2 app today with zero changes, and the same code runs offline; (b) an event stream from the
  socket server / `tds_core` phase 2 — lower latency, needs the engine. **Start with (a).** It
  limits cadence to "per saved file/scan", which matches every tool listed above.
- **Python 2 should host nothing new.** The tools need numpy/scipy versions and py3 syntax we
  already use; the py2 app's job shrinks to acquisition until `tds_core` replaces it.
- **Web stack: keep it boring.** A small Python server (e.g. FastAPI or Flask) with server-sent
  events, plain HTML, and one plotting library. No front-end framework — that would be the
  comprehension debt in another language.

---

## 3. Staging (each stage useful on its own)

| Stage | What | Needs |
|---|---|---|
| 0 (now) | Tools as pure packages with CLIs — the bench tools; water and magnet-reversal checks next | nothing new |
| 1 | One tool contract (spec + help + params + payload) unifying `@diagnostic` and `@bench_tool`; generated help pages; `diagnostics.ndjson` sidecar writer | stage 0 |
| 2 | **Read-only health page**: watches a data folder, runs enabled tools at their cadence, flags, writes the sidecar. Runs next to the py2 app | stage 1; acquisition-PC memory check |
| 3 | Provenance in every run_me bundle; then the **bundle browser** with load snippets | independent of 2; can run in parallel |
| 4 | **Control pages** (motors via REPL, setup profiles, split scan) through the socket protocol, behind the single-writer lock | stage 2 shell; bench verification |
| 5 | `tds_core` phase 2 scan engine + event stream; the web app talks to py3 core; py2 retired | `tds_core` |

Ellipsometry motors (magnet first) slot into stage 4 and remove the purge-box opening from the
innermost loop.

---

## 4. Still open

1. Confirm whether the py2 app writes a per-acquisition backup, and where (sets the health page's
   cadence).
2. After the factory reset: measure free memory with the py2 app running before relying on the
   local health page.
