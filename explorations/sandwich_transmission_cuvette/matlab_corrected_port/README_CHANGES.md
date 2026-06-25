# Corrected cuvette extraction (MATLAB) — changelog

Two-step workflow, same as the originals. Run in order, in the folder with the
`*_tr.dat` traces:

1. **`Cuvette_code_corrected.m`** — empty-cuvette → substrate (window) index `n_sub(f)`.
   Writes `Substrate_%dK_tr.dat_n_ex_real` / `_n_ex_imag` / `_freq`.
2. **`sample_code_corrected.m`** — filled vs empty cuvette → sample `n, k`, ε, σ,
   and the Drude–Smith fit. Reads the substrate files from step 1.

These are drop-in replacements for `Cuvette_code_universal.m` /
`sample_code_universal.m`. Every change is flagged inline with `% CORRECTED`.

## What changed and why

### `Cuvette_code_corrected.m` (substrate step) — had the real bugs

1. **Fabry–Pérot sign-slip (fixed).** The air-gap etalon was
   `(1 - r23.*r34.*phase3.^2)`. Because `r34 = -r23`, that evaluates to
   `(1 + r²P²)` — the wrong sign. The textbook etalon is `1/(1 - r²P²)`, which in
   this code's reflection convention is written `(1 + r23.*r34.*phase3.^2)`. The
   **sample code always had the correct sign**; only the substrate code was slipped.
   Impact is small for a gated thick window, but the sign is now correct if the
   gap etalon matters.

2. **Asymmetric zero-padding (fixed).** The original padded the two traces
   differently — reference `[150; …; 210]`, transmitted `[210; …; 150]`. That is a
   60-sample (~3 ps) **relative** time shift between reference and transmitted
   pulses, i.e. a spurious group delay baked into the transfer-function phase. It
   inflated the extracted window index (fused silica came out ~2.2–2.5 instead of
   ~1.95). Both traces are now padded **identically**, so the true inter-pulse
   delay — the real physics — is preserved.

3. **Search range widened.** `1.71–1.80` rails *below* the true fused-silica value
   (~1.95) once the padding is fixed. Widened to `1.70–2.30`. Adjust to your glass.

4. **Exact speed of light** `c = 299792458` (was `2.997925e8`).

5. **Leading-baseline removal** added before padding (matches the sample code).

### `sample_code_corrected.m` (sample step) — already correct, only annotated

- **FP sign:** already `(1 + r.*r.*phase.^2)` = `1/(1 - r²P²)`. Correct — kept,
  now commented so it isn't "fixed" by mistake.
- **Locked spacer:** the sample layer and the empty reference gap already share
  `L3` (`phase3` and `phase3ref` both use it), keeping the filled/empty paths
  symmetric. Kept and commented.
- **FP kept ON:** the thin sample/gap (~0.1 mm, ~0.9 ps round-trip) cannot be
  time-gated, so its internal reflection must be modelled. The mm-scale windows
  *are* gated, hence no window etalon term. Correct as written.
- **Exact speed of light** `c = 299792458` (was `2.997925e8`).

## Set these for your sample

- `L2`, `L4` — substrate window thicknesses (m).
- `L3` — sample/gap thickness (m); locked spacer ⇒ this is both the sample layer
  and the empty-gap thickness.
- `nmin/nmax`, `kmin/kmax` — search ranges (the substrate range must bracket your
  glass index).
- `Temps`, file-name patterns — to match your dataset.

## Validation

The corrected substrate-step physics (FP sign + symmetric padding) and the
sample-step model were cross-checked against an independent Python re-port on the
200 K triplet: substrate `n ≈ 1.95` (fused silica) and the two FP-on sample
inversions agree to ~1e-3 in n and k.
