# Why a misaligned mirror distorts a reflection lineshape

A self-contained explanation of what angular misalignment does to a THz-TDS reflection
measurement, why it produces a frequency-dependent (lineshape) distortion, and why a
delay-correction cannot undo it. Written to be explainable to others.

**Companion figure:** `explorations/reflection_phase_and_windowing/misalignment_recenter_nk_figure.png`
(raw vs recentered |H|, n, k for a gold-mirror tilt series −15 → +22.5 mrad).
**Background:** `reports/lineshape_and_inversion_tutorial.md` (the general signal-vs-artifact
framework this is a specific case of).

---

## 1. The question

We measure reflection as a ratio: `H(ω) = Y_sample(ω) / Y_reference(ω)`, then invert `H` to
n, k. If sample and reference are the *same* gold mirror, the only thing that should differ
between the two acquisitions is **nothing** — so ideally `H = 1` (flat, unity) and the inverted
index is that of a perfect mirror (n → large, r = −1). In practice, if the mirror is **tilted**
between the two acquisitions, `H` comes out as a frequency-dependent mess and n, k are nonsense.
Why does a *geometric* tilt — which you'd think is just a delay or a flat loss — create a
*frequency-dependent* distortion?

## 2. The measurement chain — what we actually detect

A THz-TDS reflection setup is, in essence:

```
emitter → [OAP collimate] → (collimated broadband beam) → SAMPLE/MIRROR
        → [OAP focus] → detection focus ⊗ gate beam → detected E(t)
```

Two facts about this chain drive everything:

1. **The detected signal is a spatial OVERLAP.** At the detection focus the THz field is
   sampled by a gate (a tightly focused optical probe in EOS, or the biased gap of a
   photoconductive antenna). What you detect at each frequency is the **overlap integral** of
   the THz focal field with that fixed, small gate spot. Move the THz focus off the gate and the
   detected amplitude drops.

2. **The THz beam is broadband and diffraction-limited, so every frequency focuses
   differently.** The focal spot size scales with wavelength, `w(ω) ∝ λ f / D ∝ f_number · λ`.
   Low frequencies (long λ) form a **large** focal spot and diverge more; high frequencies form
   a **tight** spot. This is the "cone of frequencies": a single THz pulse is a nested set of
   beams of different widths sharing the focus.

The reference acquisition and the sample acquisition each have their own overlap. `H` is the
ratio of the two — so `H` is flat and unity **only if the two acquisitions couple to the gate
identically at every frequency.**

## 3. What misalignment does to the beam — the geometry

Tilt the mirror by a small angle α. Two immediate consequences:

- **The reflected beam deflects by 2α** (law of reflection). In the collimated space this is a
  pure angular tilt of the beam.
- **At the downstream focusing optic, a tilted collimated input lands its focus off-axis.** To
  first order the focal spot is displaced transversely by `Δx ≈ f · 2α` (the parabola behaves
  like a lens: input angle → output position). The gate sits at the on-axis focus and does not
  move, so the THz focus **walks across the gate by Δx**.

Put numbers in: with a focusing OAP of `f ≈ 50–100 mm`, a tilt of `α = 7.5 mrad` gives
`Δx ≈ f · 2α ≈ 0.75–1.5 mm`. A THz focal spot is sub-mm to a few mm depending on frequency, so
**a few-mrad tilt walks the focus by about one spot — a large fractional change in overlap.**
This is why such a tiny angle is so destructive.

## 4. Why the distortion is frequency-dependent — the lineshape

Now combine §2.2 and §3. The focus walks by the *same* `Δx ≈ f·2α` at every frequency, but the
*spot size* against which that walk is measured is **not** the same:

- For **low** frequencies the spot is large, so `Δx` is a *small fraction* of the spot → the
  overlap with the gate barely changes → little amplitude loss.
- For **high** frequencies the spot is tight, so the *same* `Δx` walks it *most of the way off*
  the gate → large amplitude loss.

So the misalignment imprints a **frequency-dependent amplitude envelope** on the sample
acquisition that the reference does not have — and the ratio `H(ω)` carries exactly that
envelope. **A purely geometric tilt becomes a spectral filter** because the geometry interacts
with a frequency-dependent beam. That filter *is* the lineshape. (See the |H| column of the
figure: each tilt produces a different, smooth, frequency-shaped |H| — not a flat scaling.)

The same wavefront tilt also imprints a **frequency-dependent phase**: the off-centre, tilted
focus changes the relative arrival across the band, so the phase of `H` is not a single rigid
delay but a frequency-dependent group delay (this is the key to §5).

### The sign asymmetry (a real diagnostic)

In the data, negative tilts give **|H| > 1** (1.13 at −7.5 mrad) while positive tilts give
**|H| < 1** (0.44 at +22.5 mrad). A passive ratio of two *identical* mirrors cannot exceed 1
unless the **reference coupled worse than the sample**. So |H| > 1 is telling us the nominal
"0 mrad" reference was itself slightly off the optimum: a small *negative* sample tilt walked the
focus *closer* to ideal gate overlap than the reference, lifting the ratio above 1. This is a
genuinely useful read — it exposes that the reference alignment was not at the coupling peak, and
it is exactly why reference-based reflection is so finicky.

## 5. Amplitude vs phase, and why a delay-correction can't help

Decompose the misalignment effect on the sample spectrum into **magnitude** and **phase**:

```
Y_sample(ω) = |Y_sample(ω)| · e^{i φ(ω)}
```

A **rigid time shift** Δt — what "correcting the delay" means — multiplies the spectrum by
`e^{−iωΔt}`. That is a **pure phase factor of unit magnitude**:

```
|e^{−iωΔt}| = 1   ⟹   it cannot change |Y|, and cannot change |H|.
```

This is the mathematical heart of the result: **re-timing the pulse is invisible to |H| by
construction.** The amplitude lineshape from the frequency-dependent overlap (§4) is a real
coupling loss; it lives in `|H|`, where no delay correction can reach it.

And the phase part isn't a clean rigid delay either. The misalignment distorts the pulse *shape*
(the wavefront tilt makes the overlap frequency-dependent), so the **peak position ≠ the group
delay**. Aligning peaks removes a rigid linear-phase term but leaves a frequency-dependent
residual. A pure delay-correction therefore (a) does nothing to the amplitude error and (b) only
partially addresses the timing.

## 6. The evidence — `misalignment_recenter_nk_figure.png`

The figure runs the single-reflection pipeline twice — once raw, once after
`recenter_peaks_to_common_t0` (which resets every pulse to a common peak T0, i.e. the idealised
delay-correction) — and overlays them (solid = raw, dashed = recentered):

- **|H| column:** solid and dashed lie *exactly* on top of each other. Median |H| is identical
  to three decimals (e.g. +22.5 mrad: 0.441 raw vs 0.441 recentered). **The amplitude distortion
  is completely immune to the delay-correction**, precisely as §5 predicts.
- **n, k columns:** these *do* move between raw and recentered (they are phase-derived, and
  recentering changed the phase) — but **neither version is the physical mirror**. The numbers
  shift around without the measurement being rescued.
- **+7.5 mrad row** is the clean edge case: its peak already coincided with the reference, so
  recentering barely shifts it, yet it is still corrupted (|H| ≈ 0.82, n,k wrong). Distortion
  with ~zero delay → proof the damage is not a delay.

## 7. Why n, k go to nonsense

The inversion takes `r_sample = r_reference · H` and solves the Fresnel equation for the index.
For a gold mirror, the truth is `r = −1` (`H = 1`), which maps to `n → ∞` (a perfect conductor).
Reflection of a good conductor sits right against the unit circle `|r| ≈ 1`, where the `r → n`
map is **extremely ill-conditioned** (a small error in `r` blows up into a huge swing in n; see
the tutorial §4). So once `H` carries the misalignment's amplitude and phase errors, the inverted
n, k are not a small perturbation of "mirror" — they collapse to meaningless finite values. The
misalignment corrupts `r`; the near-unit-circle inversion amplifies that corruption into the
wild n, k you see.

## 8. The general principle — this is an instrument lineshape

Tie it back to the one diagnostic that separates signal from artifact (tutorial §7): **turn the
knob and watch.** Here the knob is the tilt angle. The lineshape's shape and magnitude **change
with the tilt** (−7.5 vs +22.5 mrad give entirely different |H|), and it **vanishes at zero
tilt**. A real material feature would be pinned by an internal energy scale and would not move.
So the misalignment lineshape is, unambiguously, an **instrument/coupling artifact** — a
property of how the beam couples to the detector, not of any sample. It is the same family as
Fabry–Pérot fringes and the gap-phase inversion spike: its "frequency content" is set by a
measurement parameter, so the measurement parameter moves it.

## 9. Practical takeaways

- **Misalignment is a frequency-dependent coupling filter, not a flat loss or a pure delay.** It
  arises because a fixed geometric beam-walk (`Δx ≈ f·2α`) is measured against a
  frequency-dependent focal spot — high frequencies suffer most.
- **You cannot post-correct it.** A delay-correction is mathematically invisible to `|H|`, and
  the residual phase is frequency-dependent. The fix is upstream: align the sample reflection to
  the *same* gate coupling as the reference (this is why the window-coupled self-referencing work
  was so sensitive to it).
- **|H| > 1 is a reference-alignment alarm.** It means the reference acquisition was below the
  coupling optimum; treat it as a flag, not a number to invert.
- **Near-mirror samples are the worst case** for showing this, because the `r → n` inversion is
  most sensitive there — but the underlying coupling distortion is present for any sample and is
  simply harder to spot when `|r|` is well away from 1.

---

*Generated alongside `run_me_reflection_single.py` (the single-bounce pipeline),
`recenter_peaks_to_common_t0` (the idealised delay-correction), and the misalignment series
`2026-06-25_misalign_tests` (gold mirror, 0 → ±22.5 mrad).*
