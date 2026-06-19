# Substrate-Sandwich THz-TDS Extraction — A Teaching Guide

*Geometry: air | substrate | sample | substrate | air*

This document explains, from first principles, how we pull the complex
refractive index `n − ik` of a thin sample out of a THz-TDS measurement when
the sample is sandwiched between two substrate windows. It covers the physics,
the model, the solver, why we make each choice, and how the three
implementations we have looked at relate to the code in `thz_core`.

It is written to *teach the method*, not to document the API. For the API, read
the docstrings in `thz_core/multilayer.py` and `thz_core/invert_grid.py`.

---

## 1. What we are actually trying to do

We shine a single-cycle THz pulse through a stack and measure the electric
field `E(t)` that comes out the other side, as a function of time. We do this
twice (at least): once through a **reference** stack and once through the
**sample** stack. The sample changes the pulse — it delays it (because the
sample is optically denser than what it replaced) and it shrinks it (because
the sample absorbs). Our job is to invert those two effects into the two
numbers we care about at each frequency:

- `n(ω)` — the **real refractive index**: how much the sample slows light.
  This shows up as a **time delay / phase shift**.
- `k(ω)` — the **extinction coefficient** (imaginary part): how much the sample
  absorbs. This shows up as an **amplitude reduction**.

We bundle them as a single complex number with a fixed sign convention (Section 9):

```
n_hat(ω) = n(ω) − i·k(ω),     k ≥ 0 means absorption
```

Everything else in this document is machinery for getting from "two measured
waveforms" to "`n_hat(ω)`" without lying to ourselves about the physics in
between.

---

## 2. First principles: where n and k hide in the waveform

### 2.1 From time to frequency

A THz pulse is broadband — one pulse contains every frequency from ~0.1 to a
few THz at once. The optical constants `n(ω), k(ω)` are frequency-dependent, so
we cannot work in the time domain directly. We Fourier transform each waveform:

```
E(t)  →  Ẽ(ω) = |Ẽ(ω)| · exp(i·φ(ω))
```

Now each frequency is a single complex number: a **magnitude** `|Ẽ|` and a
**phase** `φ`. The magnitude is "how much of this frequency survived"; the phase
is "how much this frequency was delayed."

### 2.2 The transfer function — divide out everything you don't care about

We never measure `n` from one waveform. We measure a **ratio** of two waveforms,
the **transfer function**:

```
H(ω) = Ẽ_sample(ω) / Ẽ_reference(ω)
```

This is the single most important idea in TDS. The laser pulse shape, the
emitter and detector response, the optics, the humidity in the room — all of it
is common to both measurements and **divides out**. What is left in `H(ω)` is
*only* the difference between the two stacks: the sample.

- `|H(ω)|` carries the absorption (and Fresnel losses at the new interfaces).
- `arg H(ω)` carries the delay, hence `n`.

The naive thin-slab reading is:

```
arg H(ω) ≈ −(n − 1)·ω·d / c        →   n ≈ 1 − c·arg H / (ω·d)
|H(ω)|   ≈ (Fresnel factor)·exp(−k·ω·d / c)   →   k from the leftover decay
```

The whole rest of the method is about replacing those two `≈` signs with the
*correct* multilayer expressions, because in a sandwich the "Fresnel factor" is
not the simple air-slab one, and there are extra interfaces and possibly
internal echoes.

---

## 3. The multilayer Fresnel model — the building blocks

A layered stack does only three things to a wave. Each is a small, pure
function in `thz_core/multilayer.py`.

### 3.1 Interfaces — transmission and reflection (boundary matching)

When a wave crosses from medium `j` into medium `k`, Maxwell's boundary
conditions (tangential E and H continuous) force a fixed split between the
transmitted and reflected amplitudes. At normal incidence:

```
t_jk = 2·n_j / (n_j + n_k)          (transmission, fresnel_transmission)
r_jk = (n_j − n_k) / (n_j + n_k)    (reflection,   fresnel_reflection)
```

First-principles picture: the incident field drives the charges in medium `k`;
those charges re-radiate. The ratio `n_j/n_k` sets how much of the driving field
the new medium "accepts" versus "rejects." Note `r_jk = −r_kj`: crossing the same
interface the other way flips the reflection sign. That sign bookkeeping is
exactly what trips people up by hand and exactly why we keep it in named
functions.

### 3.2 Propagation — the phase clock

Inside a homogeneous layer of thickness `d`, the wave just accumulates phase and
(if absorbing) decays:

```
P(n, ω, d) = exp(−i·(ω/c)·n·d)        (propagation_phase)
```

With `n_hat = n − ik`, this single expression does both jobs at once:

- the **real** part `n` gives the phase `exp(−i·ω·n·d/c)` → the delay,
- the **imaginary** part `−ik` gives `exp(−k·ω·d/c)` → Beer-law absorption.

The minus sign in the exponent is a **convention choice** tied to how NumPy's FFT
defines forward transform. We commit to it everywhere so that "absorption" always
means `k > 0`. (Section 9.)

### 3.3 Fabry–Pérot — the echoes you cannot always throw away

A layer with two reflecting faces is an etalon. Some light bounces back and forth
inside it before leaving. Summing that infinite geometric series of round trips
gives a closed-form correction factor:

```
FP(j,k,l) = 1 / (1 + r_jk·r_kl·P_k²)     (fabry_perot_factor)
```

where `P_k²` is one full round trip through layer `k`. **This is the term that
separates a "thin" treatment from a complete one**, and it is the crux of the
decision you just made (Section 5).

### 3.4 One layer, assembled

Putting an interface-in, a propagation, and an interface-out together gives the
transfer of a single layer embedded between two media:

```
T = t(n_before, n_layer) · P(n_layer, ω, d) · t(n_layer, n_after)
                                                  (ama_transfer_matrix)
```

If echoes are present, you multiply by the appropriate `FP` factor. If they are
gated away in time (Section 5), you omit it.

---

## 4. The sandwich geometry, specifically

Our stack is **air | substrate | sample | substrate | air**. The substrate
windows are thick and well-characterised; the sample is the thin filling. We use
two naming shorthands from the companion MATLAB code:

- **ASASA** — air | Sub | Air-gap | Sub | air. The *empty* cell. The middle "A"
  is the gap where the sample will go. Used to characterise the substrate.
- **ASMSA** — air | Sub | Material | Sub | air. The *filled* cell.

### 4.1 Why two steps, not one

The sample's effect is entangled with the substrate's. If we don't know the
substrate `n_sub(ω)` precisely, we cannot separate it from the sample. So:

**Step 1 — characterise the substrate (ASASA → n_sub).**
Measure the empty cell (or a single substrate window) against an air/empty
reference. Invert for `n_sub(ω)`. In your pipeline this is
`geometry="substrate_only"`, dispatched by `_grid_invert_substrate_only`. The
model used is `asasa_transfer_function`.

**Step 2 — characterise the sample (ASMSA → n_sample), using n_sub.**
Measure the filled cell against the empty cell. Now the substrate windows appear
*identically* in both measurements, so they cancel, and what survives is the
sample versus the air gap it replaced. This is `geometry="substrate_sandwich"`,
dispatched by `_grid_invert_substrate_sandwich`, using model
`asmsa_transfer_ratio` and feeding in the `n_sub` from Step 1.

### 4.2 The reference trick — what cancels and why

Step 2's model is a **ratio of two stacks**:

```
        T_sample      t(sub, sample)·P(sample, d)·t(sample, sub)
H  =  ───────────  =  ───────────────────────────────────────────
        T_empty       t(sub, air)·P(air, d_gap)·t(air, sub)
```

The outer substrate windows are the *same* slab of material in both numerator and
denominator, so their interface and propagation terms divide to 1 and vanish. We
are left comparing "wave through sample" to "wave through the air gap the sample
displaced." This is why the method is robust: we never need to know the substrate
*thickness* in Step 2, and the bulk of the substrate's dispersion cancels. We only
need its `n_sub` to get the two inner interfaces `t(sub, sample)` and
`t(sub, air)` right.

A good sanity check baked into the model: if `n_sample = n_air = 1` and the sample
thickness equals the gap, `H = 1` exactly. The sample "does nothing" and the
ratio knows it.

---

## 5. Fabry–Pérot: opt-in, not default — and the explicit medium

This is the decision we made together, and it deserves its reasoning recorded.

There are two honest ways to handle the internal echoes of a layer:

1. **Gate them out in time.** If the layer is thick, the first transmitted pulse
   and its first internal echo are well separated in the time-domain trace. You
   put a window around the main pulse, throw the echo away, and then you may
   legitimately drop the `FP` factor — there are no round trips left in your data.

2. **Model them.** If the layer is thin, the echo arrives *on top of* the main
   pulse — they overlap in time and you physically cannot window one without
   mangling the other. The echo is part of your signal whether you like it or
   not, so the model must include it. That means keeping the `FP` factor.

The round-trip delay inside a layer is `Δt ≈ 2·n·d/c`. Which way you go depends on
how thick the layer is relative to the pulse width.

**Our default is the gated limit (no FP), with FP available as an opt-in.** We
often run *thicker* samples whose echo is cleanly separated and gated out — that is
case 1, and it is the safe default because it leaves existing pipelines unchanged.
When a sample (or gap) is thin enough that `Δt` is a fraction of a picosecond — the
echo riding on the main pulse, ungatable — you flip `fabry_perot: True` and the
model re-introduces the etalon terms. (Earlier we considered making FP the default;
we reversed that, because the common case here is the gatable thick sample.)

**Which layers get an etalon term.** Only the *thin* layer(s) you are
characterising. The substrate slabs are always the thick, gatable element, so they
carry no `FP` term even with `fabry_perot=True`:

- **Sample step (ASMSA):** etalon on the sample layer *and* the empty reference gap.
- **Substrate step (ASASA):** etalon on the *gap* only — matching the single
  `(1 − r·r·P²)` term in the MATLAB `Cuvette_code`; the substrate echoes are gated.
- **Free-standing:** an optional slab etalon for thin free films.

```
substrate (FP on):  base · FP(n_sub, n_med, n_sub; d_gap)
sample    (FP on):  [ama_sample · FP(n_sub, n_samp, n_sub; d_s)]
                    / [ama_gap   · FP(n_sub, n_med,  n_sub; d_gap)]
```

The `fabry_perot_factor` primitive already exists in `multilayer.py`, so this is
wiring an existing building block into the geometry models, not new physics.

> **One open question, deferred to validation.** Our `fabry_perot_factor` is the
> textbook etalon `1/(1 − r²P²)` (with resonant poles). The MATLAB `Cuvette_code`
> writes the gap denominator as `(1 − r23·r34·P²)`, which with `r34 = −r23` is
> `(1 + r23²P²)` — the *opposite* sign. That is a real convention discrepancy, not
> something to wave away by derivation. We use the textbook form and resolve it
> numerically against the MATLAB output on real data (Section 8, step 2).

### 5.1 The surrounding medium: vacuum vs air

The model also needs to know what is *outside* the solid — the ambient and the
empty reference gap. We expose this as one `medium_index` parameter (default `1.0`,
vacuum). If you measure in dry air, pass `≈1.00027`.

Why bother, when air is "basically vacuum"? Because the medium fills the **empty
reference gap**, and that term does *not* fully cancel in the filled/empty ratio.
The leftover is a small but **systematic** phase, `(ω/c)·(n_med − 1)·d_gap` — about
`0.017 rad` for a 1 mm gap at 3 THz. It biases the extracted index in one
direction. Making the medium an explicit number documents the assumption instead of
burying a hidden `1` in the math. (Measurements taken "in air" that are really under
vacuum we treat as vacuum for now; dispersive air is a later refinement.)

---

## 6. Solving the inverse problem

We have a measured `H(ω)` and a model `H_model(n_hat; ω)`. We want the `n_hat`
that makes them match at each frequency. Two routes.

### 6.1 Why it is not simply closed-form

For a free-standing slab in air you *can* solve it almost directly: phase → `n`,
log-amplitude → `k`, with a couple of Fresnel-correction iterations. That is what
`invert_nk` (the analytic inverter) does, and what the **Novelli closed-form**
does for the sandwich. It is fast and elegant.

But once you keep Fabry–Pérot, `n_hat` appears inside `r`, inside `t`, *and*
inside `P²` in a denominator. The equation `H_model(n_hat) = H_meas` becomes
transcendental — no clean algebraic inverse. So for the full-FP model we fall
back to a numerical search.

### 6.2 The grid search

`invert_nk_grid` does the brute-force-but-honest thing: at each frequency it lays
down a dense grid of candidate `n_hat = n − ik` values, evaluates `H_model` at
every one (vectorised — this is why the model functions take arrays), and picks
the candidate whose model output is closest to the measured `H`. The residual is
an L1 distance in the complex plane:

```
residual = |Re(H_model − H_meas)| + |Im(H_model − H_meas)|
```

Accuracy is set by the grid step (`±step/2`); halving the step quadruples the
runtime. Coarse for surveys, fine for final numbers.

### 6.3 The branch problem and continuity tracking

Here is the subtle failure mode. The model is **periodic in phase** — a delay of
`φ` and a delay of `φ + 2π` look identical at a single frequency. So at any one
bin there can be several `n` values that fit almost equally well (different
"branches"). Pick the global minimum independently at each frequency and you can
**hop between branches** from one bin to the next, producing a jagged `n(ω)` with
unphysical jumps.

The fix the MATLAB code uses, and the one `thz_core` now has: **continuity
tracking** (`_select_by_continuity`, opt-in via `continuity_tracking.enabled`).
Solve the first (most reliable) bin, then for every subsequent bin search only a
*small window around the previous bin's answer*. Physically this encodes the fact
that `n(ω)` is smooth — it cannot teleport to a different branch between adjacent
frequencies.

(The default `_select_best_candidate` still takes the global minimum with a weaker
"closeness to an analytic guess" tiebreaker — its own docstring notes that this can
override the correct minimum. Continuity tracking replaces that with something
physically grounded, and is the recommended mode once Fabry–Pérot is on.)

### 6.4 The analytic closed-form as a seed

The Novelli analytic formula is fast and, while it has no Fabry–Pérot, it gives an
*excellent first guess*. The natural synthesis: use the analytic answer to seed
bin 0 of the continuity search, then let the FP-aware grid refine and track from
there. Fast where it can be, rigorous where it must be.

---

## 7. The three implementations, and how ours compares

We scoped three codebases solving this same problem.

| | MATLAB (IMDEA / Vasilis) | Legacy Python (Novelli OE.510393) | thz_core (this build) |
|---|---|---|---|
| Solver | grid + **continuity tracking** | analytic closed-form | grid; global-min **or continuity (opt-in)** |
| Fabry–Pérot | **kept** | none (thin-film) | **opt-in** (gated default) |
| Surround / gap | air, explicit gap | n/a | **explicit `medium_index`**; gap modelled when FP on |
| Workflow | two-step (sub then sample) | one-step (needs known n_sub) | two-step |
| Conditioning | differential ΔE/E | direct | direct |
| Architecture | monolithic script | flat script | **layered math + thin adapter** |
| Speed | slow | **fast** | slow |

The key finding: **all three share the same Fresnel physics**; they differ in
solver, in whether they keep FP, and in packaging. Your `thz_core` already has the
cleanest architecture and the full two-step sandwich wired end-to-end through the
dataset adapter (`thz_adapter.invert_nk_grid` → three geometry helpers). It is the
right home. The two physics features it lacked relative to the MATLAB — **FP** and
**continuity tracking** — plus an explicit **surrounding medium**, are now
implemented (Section 8). The only MATLAB capability still optional/unported is the
Novelli analytic path as a fast seed. Nothing needed re-porting.

---

## 8. The build plan

In priority order:

1. **Continuity tracking + Fabry–Pérot + explicit medium.** ✅ **Done (2026-06-19).**
   - Added `FP` and `medium_index` to `asasa_transfer_function` and
     `asmsa_transfer_ratio` in the math layer, using the existing
     `fabry_perot_factor` primitive. FP is **opt-in** (`fabry_perot: False` default —
     our reversed decision, Section 5); the gated limit is the default and the
     thin-layer FP model is one config flag away.
   - Added `_select_by_continuity` to `invert_nk_grid`: a local search window around
     the previous bin's solution, seeded by the analytic guess at the first bin.
     Opt-in via `continuity_tracking.enabled` (default off = global minimum).
   - No adapter change was needed — `_grid_invert_*` already forward `config`
     verbatim, so the new keys flow straight through. Single source of defaults =
     the core; the config carries only deliberate overrides.
   - Tests: 196 thz_core tests pass (+16 new), including bit-exact regression guards
     that the defaults reproduce the previous output. See `ANALYSIS_NOTES.md` §19.

   The config block (everything folds into one `invert_grid` dict):

   ```python
   config["invert_grid"] = {
       "geometry": "substrate_sandwich",
       # physics model (defaults shown = previous behaviour)
       "medium_index": 1.0,        # 1.0 vacuum, ~1.00027 dry air
       "fabry_perot": False,       # opt-in internal echoes
       # solver
       "continuity_tracking": {"enabled": False, "search_half_width_cells": 15},
       # geometry-specific
       "thickness_sample_m": 200e-6,
       "thickness_ref_gap_m": 200e-6,
       # "thickness_gap_m": 120e-6,   # ASASA, required only when fabry_perot=True
   }
   ```

2. **Validate on real data.** ← *next*
   - The `Vasilis_Data/data` triplet (`reference_200K`, `Substrate_200K`,
     `Sample_200K`) is a one-temperature instance of exactly this workflow.
     Acceptance test: reproduce the MATLAB's `n_sub(ω)` and sample `n(ω)` on that
     triplet to within the grid resolution. This is also where the FP sign
     convention (Section 5) gets settled.

3. **Pause and assess** before going further (conductivity / Drude–Smith, the
   temperature loop, etc.).

A note on conductivity: the MATLAB's downstream `σ₁ = ω·ε₀·ε₂` and the
Drude–Smith fit are *not* inversion work. Once `n,k` are out, they are a
straightforward call into the existing `thz_core/fitting/` module — a later,
separate task.

---

## 9. Sign-convention and units cheat-sheet

Keep this pinned; most THz-TDS bugs are sign/branch bugs.

- **Complex index:** `n_hat = n − i·k`, with `k ≥ 0` for absorption.
- **Propagation:** `exp(−i·(ω/c)·n_hat·d)`. The leading minus matches the NumPy
  FFT forward-transform convention used throughout the FFT and inversion code.
- **Phase → n:** for a thin slab, `n ≈ 1 − c·arg H / (ω·d)`. A denser sample
  delays the pulse → more negative `arg H` → larger `n`.
- **Amplitude → k:** `k` comes from the residual `|H|` decay *after* dividing out
  the Fresnel/FP transmission factor — never from raw `|H|` alone.
- **Interface reflection flips sign on reversal:** `r_jk = −r_kj`. Get this wrong
  and the Fabry–Pérot factor changes sign and `n` drifts.
- **Frequency vs angular frequency:** `ω = 2π·f`. The models work in `ω`; the
  dataset axis is usually `f` in Hz. Convert once, at the boundary.
- **Thickness units:** SI metres, paired with `c` in m/s. If `f` is in Hz, `d`
  must be in metres.

---

*Companion code:* `thz_core/multilayer.py` (Fresnel primitives + geometry
models), `thz_core/invert_grid.py` (grid solver), `thz_core/invert.py` (analytic
slab + reflection inverters), `dataset_core/adapters/thz_adapter.py` (dataset
bridge). *Decision rationale and dated status* live in `ANALYSIS_NOTES.md`; this
file is the conceptual/teaching companion to those.
