# The quantity registry

`dataset_core/adapters/quantity_registry.py`

The quantity registry is the **single source of truth for every displayable/exportable
result** in the THz-TDS pipeline. A "quantity" is one scalar series versus frequency —
`n`, `k`, `|H|`, `σ₁`, the FFT magnitude, and so on. Each is registered **once**, together
with the recipe for pulling it out of a sample's `processing_dict`. Everything that shows or
writes results — the results viewer, the CSV exporter, and the `display` plotting adapter —
consumes this one registry, so **display and export can never drift apart**.

> **The payoff:** adding a new plottable/exportable quantity is a single `register(...)`
> call in this file. You never touch the viewer, the exporter, or `display.py`.

---

## 1. Mental model

The pipeline stores its results as plain arrays inside each sample object's
`processing_dict` (`fft_freq`, `n`, `k`, `eps`, `sigma`, `transfer_H`, the SNR masks, the
Monte-Carlo `*_sigma` error arrays, …). Those keys are an **internal, ad-hoc contract** —
raw NumPy arrays with no metadata about how to display them.

The registry is the thin, declarative layer on top: it says, for each named quantity,

- **where** its data lives (an `extract` function over `processing_dict`),
- **what** it is (human label, axis label, group, linear/log scale),
- **which** companion arrays go with it (the SNR/trusted-band mask, the error array),
- **whether** it applies to reference traces as well as samples,
- **how** it appears in a CSV (its column header), and
- optionally, a **model overlay** to draw on top (e.g. a Drude fit curve).

A consumer then iterates the registry and treats every quantity uniformly. It never needs a
hard-coded list of "the things we can plot."

```
processing_dict (raw arrays)  ──extract()──►  (freq_hz, y)  ──►  viewer / export / display
        ▲                                          ▲
        │                                          │
   pipeline stages write it              registry describes it (ONCE)
```

**One quantity = one line = one CSV column.** Complex results (`H`, `eps`, `sigma`, the FFT
spectrum) are *not* registered as one complex quantity; they are split into two real
quantities (real/imag, or mag/phase) so that display and export stay uniform. That is why
you see `sigma_real` and `sigma_imag`, not `sigma`.

---

## 2. The `Quantity` dataclass, field by field

```python
@dataclass
class Quantity:
    name: str                     # registry key + default export column stem, e.g. "sigma_real"
    label: str                    # human name for the viewer selector, e.g. "sigma_1 (real conductivity)"
    group: str                    # grouping for the viewer, e.g. "Conductivity"
    y_label: str                  # y-axis label, e.g. "sigma_1 (S/m)"
    extract: Callable[[dict], tuple | None]   # processing_dict -> (freq_hz, y)  OR  None
    mask_key: str | None = "transfer_mask"    # processing_dict key for the trusted-band mask
    error_key: str | None = None              # processing_dict key for the per-bin error array
    yscale: str = "linear"                    # "linear" | "log"
    include_references: bool = False          # also show this for reference traces?
    export_header: str | None = None          # CSV column name; None => NOT exported
    overlay: Callable[[dict], tuple | None] | None = None   # optional model curve (freq_hz, y)
```

| Field | What it controls | Notes |
|---|---|---|
| `name` | The registry key. Must be unique. | Used by `display.plot_quantity(dataset, name)` and `get_series`. |
| `label` | Display name in the viewer's selector. | Free text. |
| `group` | Logical grouping in the viewer. | e.g. `"Spectrum"`, `"Transfer function"`, `"Optical constants"`, `"Permittivity"`, `"Conductivity"`. |
| `y_label` | y-axis text. | |
| `extract` | **The core.** `processing_dict -> (freq_hz, y_real_array)`, or `None` when inputs are absent. | Must return a **real** y array. Return `None` (don't raise) when a needed key is missing — that's how consumers skip a quantity for a sample that doesn't have it. |
| `mask_key` | Which `processing_dict` key holds the boolean trusted-band / SNR mask for shading and for scatter masking. | Defaults to `"transfer_mask"`; FFT quantities use `"snr_mask"`; set `None` for no mask. |
| `error_key` | Which `processing_dict` key holds the per-bin uncertainty for error bars. | `None` = no error bars. These are the `*_sigma` arrays written by `uncertainty.propagate_uncertainty`. |
| `yscale` | `"linear"` or `"log"`. | FFT magnitude is `"log"`. |
| `include_references` | If `True`, the quantity is meaningful for reference traces too (the FFT spectra). If `False`, only samples. | n/k/σ are sample-only; the reference-vs-reference inversion is a singularity. |
| `export_header` | CSV column name. `None` ⇒ the quantity is **display-only**, never written to CSV. | The exporter builds its columns from exactly the quantities that set this. |
| `overlay` | Optional `processing_dict -> (freq_hz, y)` returning a **model** curve to draw on top of the data (e.g. a fit). Returns `None` when there's no model. | See §6. |

### The `extract` contract in detail

```python
extract(processing_dict) -> (freq_hz, y)   # both 1-D real arrays, same length
extract(processing_dict) -> None           # inputs missing -> skip this quantity here
```

- `freq_hz` is in **Hz** (the raw `fft_freq`). Consumers convert to THz for display.
- `y` is **real**. Complex arrays are exposed via separate real/imag (or mag/phase)
  quantities.
- Guard every access and return `None` when a required key is absent, rather than raising —
  `available_names()` and the consumers rely on `None` to mean "not available for this
  sample."

---

## 3. Public API

```python
from dataset_core.adapters import quantity_registry as registry

registry.register(quantity)            # add/replace a Quantity (idempotent by name); returns it
registry.QUANTITY_REGISTRY             # dict[name -> Quantity] (registration order preserved)
registry.display_quantities()          # list[Quantity], all of them, in registration order
registry.export_quantities_list()      # list[Quantity] with export_header set (the CSV columns)
registry.available_names(processing_dict)   # list[str]: quantities whose extract yields data here
```

- **`register`** is idempotent by `name`: registering the same name again replaces it. That
  is how you'd override a built-in.
- **`available_names`** is the "what can I plot for *this* sample?" query — it runs every
  `extract` and keeps the ones that don't return `None` (exceptions are swallowed and treated
  as "not available"). `display.available_quantities(dataset)` wraps this across a dataset.

---

## 4. Extraction helpers (don't hand-roll the common cases)

The module ships small factories so most quantities are one line:

```python
_real_series(value_key, transform=lambda a: a)
    # y = transform(processing_dict[value_key]); x = fft_freq. For a plain real array.

_complex_component(value_key, component)   # component in {"real","imag","abs","angle"}
    # y = np.real/np.imag/np.abs/np.angle(processing_dict[value_key]); x = fft_freq.
```

Both return `None` automatically when `fft_freq` or `value_key` is missing. Use
`_real_series` for arrays that are already real (`n`, `k`), and `_complex_component` for one
part of a complex array (`eps`, `sigma`, `transfer_H`, `fft_spectrum`).

For anything custom (e.g. unwrapping phase), pass a plain lambda — see `fft_phase` below.

---

## 5. How to add a quantity — recipes

All of these go at the bottom of `quantity_registry.py`, next to the existing `register(...)`
calls.

### 5a. A real array already in `processing_dict`

```python
register(Quantity(
    name="n", label="n (refractive index)", group="Optical constants", y_label="n",
    extract=_real_series("n"),          # reads processing_dict["n"]
    error_key="n_sigma",                # error bars from the MC array
    export_header="n",                  # -> CSV column "n"
))
```

### 5b. One component of a complex array

```python
register(Quantity(
    name="sigma_real", label="sigma_1 (real conductivity)", group="Conductivity",
    y_label="sigma_1 (S/m)",
    extract=_complex_component("sigma", "real"),   # np.real(processing_dict["sigma"])
    error_key="sigma_real_sigma",
    export_header="sigma_real",
))
```

### 5c. A transformed / derived quantity (custom lambda)

`extract` is just a function, so any computation is fair game. Example: absorption
coefficient `α = 2 ω k / c = 4π f k / c` (per metre), computed on the fly from `k`:

```python
import numpy as np
_C_M_PER_S = 2.99792458e8

def _extract_absorption(processing_dict):
    freq = processing_dict.get("fft_freq")
    k = processing_dict.get("k")
    if freq is None or k is None:
        return None
    alpha = 4.0 * np.pi * np.asarray(freq) * np.asarray(k) / _C_M_PER_S
    return np.asarray(freq), alpha

register(Quantity(
    name="absorption", label="absorption coefficient", group="Optical constants",
    y_label="alpha (1/m)", extract=_extract_absorption,
    error_key=None, export_header="absorption",
))
```

That single call makes `absorption` appear in the viewer selector, plottable via
`display.plot_quantity(dataset, "absorption")`, and a new CSV column — with no other edits.

### 5d. Phase (needs unwrapping) — the lambda pattern used by the built-ins

```python
register(Quantity(
    name="transfer_phase", label="Transfer phase", group="Transfer function",
    y_label="arg H (rad)",
    extract=lambda pd: (None if pd.get("transfer_H") is None or pd.get("fft_freq") is None
                        else (np.asarray(pd["fft_freq"]), np.unwrap(np.angle(pd["transfer_H"])))),
    mask_key="transfer_mask", error_key="transfer_phase_sigma", export_header="transfer_phase",
))
```

### Checklist when adding a quantity

- [ ] `extract` returns `(freq_hz, real_y)` or `None`; never raises on missing keys.
- [ ] `error_key` points at a real array **on the same frequency grid** as `fft_freq`
      (the resolution decimation keeps them aligned — see §7), or is `None`.
- [ ] `mask_key` is right (`"snr_mask"` for spectra, `"transfer_mask"` for transfer/derived,
      `None` for unmasked).
- [ ] `include_references=True` only if the quantity means something for a reference trace.
- [ ] `export_header` set if you want a CSV column; leave `None` for display-only.
- [ ] Add/extend a test in `tests/test_quantity_registry.py` (extraction) and, if it's
      plottable, it's already covered generically by `tests/test_display.py`.

---

## 6. Overlays (model curves, e.g. fits)

`overlay` lets a quantity draw a **model** on top of the measured points — used for fit
curves. It has the same signature as `extract` (`processing_dict -> (freq_hz, y)` or `None`)
but returns the *model* evaluated on the frequency axis.

The built-in conductivity overlay (`_conductivity_fit_component`) reads a stored `fit_result`
from `processing_dict`, looks the model up in `thz_core`'s `MODEL_REGISTRY`, evaluates it over
`fft_freq`, and returns the real or imaginary part. It returns `None` when there's no
successful fit, so the overlay simply doesn't draw. This is why `sigma_real`/`sigma_imag`
show a Drude curve once you've fit, and nothing before.

To give a new quantity an overlay, pass `overlay=...`. To hook a **new fit model** into the
existing conductivity overlay, you generally register the model in `thz_core`'s
`MODEL_REGISTRY` (target `"sigma"`) rather than touching this file — the overlay evaluates
whatever model the `fit_result` names.

---

## 7. Who consumes the registry (and one alignment guarantee)

- **`display.py`** (the plotting adapter) — `plot_quantity`, `get_series`,
  `available_quantities` all look quantities up by `name` and read `extract`, `mask_key`,
  `error_key`, `yscale`, `y_label`, `overlay`. See `docs/` / the module docstring.
- **`results_viewer.py`** — the interactive viewer builds its selector from
  `display_quantities()` grouped by `group`, and draws using the same fields.
- **`results_viewer.export_quantities`** — the CSV exporter builds its columns from
  `export_quantities_list()`; a sample missing a quantity gets a NaN column, so every CSV has
  the same shape. By default it also writes `fit_summary.csv` and `README.md` (see below) so the
  export directory is self-describing — pass `readme=False` to skip them.
- **`results_viewer._write_export_readme`** — writes the export directory's `README.md`. Its
  column glossary is built from `export_quantities_list()`'s `label`/`y_label`, so a newly
  registered quantity documents itself with no second list to maintain. Also lists each fitted
  sample's model, parameters, and derived scattering-rate/crossover readouts (see
  `results_viewer.export_fit_summary` below).

### Handing an export to a collaborator

`export_quantities` is meant to be handed off, not just consumed by other code in this repo. By
default (`readme=True`) it writes two extra files alongside the per-sample CSVs:

- **`fit_summary.csv`** — one row per sample that carries a successful `fit_result`: the model
  name, every parameter with its 1-sigma uncertainty and unit (from `FitResult.param_units`),
  R^2/reduced chi^2/RMSE, and — for any tau-bearing model (Drude, Drude-Smith) — the derived
  `scattering_rate_Hz` (`1/tau`) and `crossover_THz` (the sigma_1 = sigma_2 crossover frequency).
  Superset-safe across models: a Drude-Smith `c` column is simply blank on plain-Drude rows.
- **`README.md`** — generation timestamp, source dataset path, a plain-English column glossary
  for `<sample>_results.csv` (read straight off the registry, so it can't drift from what's
  actually in the CSVs), and each fitted sample's readout (`FitResult.summary_lines()` plus the
  scattering-rate/crossover lines).

Use `thz.export_fit_summary(dataset, export_dir)` directly if you only want the fit table (e.g.
to re-export it after adding a fit without re-writing every per-sample CSV).

**Alignment guarantee for `error_key` / `mask_key`:** the pipeline's
`apply_instrument_resolution` decimates *every* 1-D array in `processing_dict` whose length
matches `fft_freq` — values, masks, and `*_sigma` errors alike — onto the true
instrument-resolution grid. So a correctly-written quantity's value, mask, and error arrays
stay the same length and co-indexed automatically; you don't manage that yourself.

---

## 8. Built-in quantities (current)

| `name` | group | y-axis | source key | mask | error | log? | refs? | CSV |
|---|---|---|---|---|---|---|---|---|
| `fft_mag` | Spectrum | `\|FFT\|` | `fft_spectrum` (abs) | `snr_mask` | — | yes | yes | ✓ |
| `fft_phase` | Spectrum | `arg FFT (rad)` | `fft_spectrum` (unwrapped angle) | `snr_mask` | — | — | yes | ✓ |
| `transfer_mag` | Transfer function | `\|H\|` | `transfer_H` (abs) | `transfer_mask` | `transfer_H_sigma` | — | — | ✓ |
| `transfer_phase` | Transfer function | `arg H (rad)` | `transfer_H` (unwrapped angle) | `transfer_mask` | `transfer_phase_sigma` | — | — | ✓ |
| `n` | Optical constants | `n` | `n` | `transfer_mask` | `n_sigma` | — | — | ✓ |
| `k` | Optical constants | `k` | `k` | `transfer_mask` | `k_sigma` | — | — | ✓ |
| `eps_real` | Permittivity | `eps_1` | `eps` (real) | `transfer_mask` | `eps_real_sigma` | — | — | ✓ |
| `eps_imag` | Permittivity | `eps_2` | `eps` (imag) | `transfer_mask` | `eps_imag_sigma` | — | — | ✓ |
| `sigma_real` | Conductivity | `sigma_1 (S/m)` | `sigma` (real) | `transfer_mask` | `sigma_real_sigma` | — | — | ✓ (+ fit overlay) |
| `sigma_imag` | Conductivity | `sigma_2 (S/m)` | `sigma` (imag) | `transfer_mask` | `sigma_imag_sigma` | — | — | ✓ (+ fit overlay) |

---

## 9. Design rationale & gotchas

- **Why split complex into two quantities?** So the whole system stays "one series = one
  line = one column." A single complex quantity would force every consumer to special-case
  real/imag, which is exactly the drift the registry exists to prevent.
- **Return `None`, never raise, in `extract`.** `available_names` and the consumers treat
  `None` as "not present." Raising would either crash a consumer or (in `available_names`) be
  silently swallowed anyway — returning `None` is the explicit, correct signal.
- **`export_header=None` is a real choice, not an oversight** — it's how you make something
  display-only (e.g. a diagnostic you don't want in the results CSV).
- **`error_key` must be a real array on the `fft_freq` grid.** For a complex quantity, that
  means a real-valued sigma for the specific component (`sigma_real_sigma`), not a complex
  error.
- **Overriding a built-in** is just `register(Quantity(name="n", ...))` again — same name
  replaces. Do this deliberately; it's global.
- **Registration order is display order.** The viewer selector and CSV column order follow
  the order of the `register(...)` calls.
```
