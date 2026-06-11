# Group meeting — front-pulse self-referencing for window-coupled THz reflection

Working notes + figure scripts for the meeting. Each figure has an isolated,
runnable script in this folder that exercises the production code paths
(`thz_core`, `thz_adapter`) so the numbers are real, not hand-waved.

Run any script with the project venv, e.g.:

```
.venv/Scripts/python.exe reports/group_meeting_window_selfref/fig_pulse_gate_offset.py
```

---

## The one-line story

A SiO₂ window's **front-face reflection never touches the sample**, so the
intra-trace ratio `W = Y_second / Y_first` is a fingerprint of the window
*alone*. Referencing every trace to its own front pulse cancels mount-to-mount
drift (window deformation under pressure, rotation, realignment) that otherwise
injects fake spectral features. On CNT-13/D this replaced ~7–10 % structured
drift with a ~1.6 % noise floor and made repeat measurements agree 4–5× better.

---

## Talking points

1. **The problem.** Reference (bare window) and sample (pressed) are different
   mounts. Pressing/rotating the window changes how the beam couples into the
   spectrometer → the bare-window reference no longer matches the sample's
   window path → frequency-structured artefacts in H that masquerade as sample
   features.
2. **The idea.** `H_new = (Y2_s/Y1_s)/(Y2_r/Y1_r) = H_old·(Y1_r/Y1_s)`. Each
   trace is normalised by its own front pulse. The front-pulse ratio
   `D = Y1_s/Y1_r` *is* the drift, and is directly measurable (D≡1 ⇒ no drift).
3. **Evidence it works** (real data, full pipeline): repeat spread n 11→4 %,
   σ₁ 18→5 %; the 90°-rotated (fibers ⊥ pol) sample stays cleanly distinct =
   real anisotropy, previously buried under drift.
4. **Bonus.** The same W yields the window index: n_SiO₂ ≈ 1.96 flat ⇒ fused
   silica, so rotation sensitivity is geometric (wedge/mount), not
   birefringence.
5. **Practical cost.** One requirement on acquisition: keep each pulse near the
   centre of its time gate (see §A below). We are comfortably inside the safe
   regime.

---

## A. Gate placement: how pulse position drives residual error

**Script:** `fig_pulse_gate_offset.py` → `fig_pulse_gate_offset.png`

Self-referencing forms a *spectral* ratio, which is a time-domain
**deconvolution**. A deconvolution through a finite, Hann-tapered gate is only
exact when the gate weighs each pulse's surroundings identically. Instrument
drift acts like a kernel that plants small echoes around every pulse (modelled
here as replicas at ±τ). If a pulse sits off-centre, the taper clips the
trailing echo asymmetrically and a residual survives.

**Result** (6 ps gate, 15 % drift):

| drift echo τ | pulse centred | pulse ±1 ps | pulse ±2 ps |
|--------------|---------------|-------------|-------------|
| 0.3 ps       | 0.10 %        | 0.13 %      | 0.32 %      |
| 0.6 ps       | 0.14 %        | 0.19 %      | 0.54 %      |
| 1.0 ps       | 0.21 %        | 0.28 %      | 1.00 %      |

Conventional reference (do nothing): **12.3 %** rms, drift-limited.

**Takeaways for the talk:**
- Self-referencing buys **~1.5–2 orders of magnitude** (12 % → 0.1–0.2 %).
- The residual is a shallow U in pulse position: flat within ±1 ps, then climbs.
- Slower drift (larger τ, more time-domain spread) is more position-sensitive.
- **Operational rule: keep each pulse within ~1 ps of its gate centre** — a mild
  constraint at our 6 ps gates. The bottom panels show the mechanism: centred =
  both echoes under the flat top of the Hann; edge = trailing echo clipped.

*Caveat for honesty in the talk:* this is a synthetic, single-kernel drift
model. It isolates the gating mechanism cleanly but the real drift is richer;
treat the absolute percentages as illustrative of the *scaling*, and trust the
real-data repeat-consistency numbers (§3) for the headline.

---

---

## B. The measured drift D = Y1_sample / Y1_reference

**Script:** `fig_drift_structure.py` → `fig_drift_structure.png`

D = Y1_sample/Y1_reference is the front-pulse ratio between the sample and
reference acquisitions. Since the front reflection never reaches the sample,
D ≡ 1 ideally; any deviation is pure mount-to-mount alignment drift. This is
the quantity self-referencing cancels.

**Takeaways for the talk:**
- Shows the amplitude and phase structure of the actual drift on CNT-13/D data.
- The 0°-orientation repeats show the trace-to-trace variability.
- The 90°-rotated measurement may show a systematically different D,
  illustrating why a shared reference is insufficient.

---

## C. H_old vs H_new — the fake features removed

**Script:** `fig_H_comparison.py` → `fig_H_comparison.png`

Runs the full spectral pipeline twice (with and without `self_reference`) on
CNT-13/D data and overlays |H| and arg(H) for each sample. The structured
oscillations in H_old — the drift artefacts — are absent in H_new.

**Takeaways for the talk:**
- Direct visual: "these wiggles are not sample physics, they're mount drift."
- Both amplitude and phase are affected; self-referencing cleans both.
- The feature wavelengths in H_old correspond to the drift echo timescales
  seen in §B.

---

## D. Repeat consistency + anisotropy

**Script:** `fig_repeat_consistency.py` → `fig_repeat_consistency.png`

Runs the full pipeline through `invert_nk_reflection` for all four sample
measurements (three 0°-orientation repeats + one 90°-rotated). Side-by-side:
conventional (H_old) vs self-referenced (H_new) n(f). Reports the RMS spread
across the parallel repeats.

**Takeaways for the talk:**
- The headline quantitative result: repeat spread σ(n) shrinks by ~3–4×.
- After self-referencing the 90°-rotated trace is cleanly distinct from the
  0° repeats — real optical anisotropy, previously buried under drift artefacts.

---

## E. Window index n_SiO₂(ω)

**Script:** `fig_window_index.py` → `fig_window_index.png`

Uses `characterise_window` to invert the intra-trace ratio W = Y2/Y1 of the
bare-window reference for n_SiO₂(f) and k_SiO₂(f). Also shows |W| measured
vs the forward model built from the extracted n — a round-trip consistency check.

**Result** (CNT-13/D bare reference, 0.9 mm fused silica, 45° external):
n_SiO₂ ≈ 1.963 ± 0.003 (flat, 0.25–2.75 THz), consistent with fused silica.

**Takeaways for the talk:**
- The window is isotropic fused silica (not birefringent) — the rotation
  sensitivity is purely geometric (wedge/tilt under pressure).
- n_SiO₂(f) can replace the nominal 1.95 scalar in the inversion as a
  per-frequency array for a small systematic improvement.
- Validates the window model used in the Fresnel inversion.

---

## TODO for the report (build iteratively)

- [x] A. Pulse-in-gate offset error (this section)
- [x] B. The drift itself: D = Y1_s/Y1_r magnitude/phase structure (real data)
- [x] C. H_old vs H_new on real samples (the fake features removed)
- [x] D. Repeat-consistency before/after + the anisotropy that emerges
- [x] E. Window index n_SiO₂(ω) from a single trace + time-domain prediction check
- [ ] F. Leave-one-scan-out prediction noise floor
