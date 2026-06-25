# Lineshapes in THz reflection: what makes them, what moves them, and how to tell signal from artifact

A working tutorial, grounded in the CNT-on-SiO₂ reflection work. The aim is one durable
instinct: **a real material feature is pinned by an internal energy scale; a measurement
artifact is pinned by a knob you control.** Turn the knob and watch.

Sign convention throughout (matches `thz_core`): `n̂ = n − ik`, `k ≥ 0`, time factor `e^{−iωt}`.

---

## 1. The chain of causation (and why distortions can enter at every link)

```
microscopic charge response  →  ε(ω)  →  n̂ = √ε = n − ik  →  r(ω) Fresnel  →  measured field
        (Drude / Lorentz)        permittivity   complex index     reflection        + window, gap,
                                                                                       timing, noise
```

We *measure* the right-hand side and *want* the left-hand side, so we run the chain
**backwards**: `r → n̂ → ε → σ`. Two facts drive everything below:

1. Each forward link has a characteristic **lineshape** set by physics (Sections 2–3).
2. The **backward** link `r → n̂` is a nonlinear map that is well-behaved in some regions
   and nearly singular in others — so it can *manufacture* features that were never in the
   material (Section 4). Add geometry (gap, window, timing) and you get more (Sections 5–6).

The diagnostic that separates these is the same every time (Section 7).

---

## 2. The Drude lineshape — free carriers (what CNT *should* be)

**Microscopic picture.** Free electrons are pushed by the field and randomized by collisions
at rate `γ = 1/τ`. There is **no restoring force** — nothing defines a special frequency. A
free charge driven at frequency ω just slides 90° out of phase as ω grows; the only scale in
the problem is the scattering rate `1/τ` and the carrier density (through the plasma frequency
`ω_p`).

**Math.**
```
σ(ω) = σ₀ / (1 − iωτ),         σ₀ = ε₀ ω_p² τ        (conductivity)
ε(ω) = ε∞ − ω_p² / (ω² + iω/τ)                        (permittivity)
```

**Lineshape and its tell.** `Re σ(ω)` is a Lorentzian **centred at ω = 0** — the "Drude peak"
sits at zero frequency, with half-width `1/τ`. There is **no peak at finite ω**. In the THz
band you see `Re σ` high and *falling* with frequency, `k ≫ n` (metallic), both monotonic.

> **The pin:** the Drude feature lives at ω = 0 because there is no restoring force. Change
> carrier density and σ₀ scales; change scattering and the width changes — but nothing slides
> a peak to a finite frequency. A metallic CNT film that shows a *finite-frequency* peak in σ
> is telling you something else is going on (a real resonance, or — our case — an artifact).

(*Drude–Smith* adds a back-scattering "persistence" `c` that can push the σ peak slightly off
zero — a common dodge for nanostructured/confined carriers. We deliberately dropped it: if the
bare Drude doesn't fit, you want to know *why*, not paper over it with an extra knob.)

---

## 3. The Lorentz lineshape — a bound resonance (phonon, exciton, intersubband, …)

**Microscopic picture.** Now the charge is **bound**: a restoring force gives a natural
frequency `ω₀` (a spring). Drive it and you get resonance — large response near `ω₀`, damped by
`γ`. The restoring force is an internal material property (bond stiffness, lattice, confinement),
so `ω₀` is a **material constant**.

**Math (one oscillator).**
```
ε(ω) = ε∞ + Δε · ω₀² / (ω₀² − ω² − iγω)
```

**Lineshape and its tells.**
- **k (absorption) PEAKS at ω₀.** Energy is absorbed on resonance — that's the definition.
- **n is dispersive through ω₀:** it rises below ω₀ ("normal" dispersion), then drops sharply
  across ω₀ ("anomalous" dispersion), then recovers. The n wiggle is the Kramers–Kronig partner
  of the k peak — they are not independent.

> **The pin:** `ω₀` is fixed by the restoring force. It does **not** move if you change the
> gap, the window, the alignment, or the inversion assumptions. *That immovability is the
> single most reliable signature of a real resonance.*

This is exactly the test we ran on the CNT "Lorentzian" — and it failed it (Section 7).

A subtle but decisive point you just internalised: **n-peak with a k-peak ⇒ resonance; n-peak
with a k-DIP ⇒ not a resonance.** Real absorption puts the action in k. If n spikes where k
*notches down*, the feature is not coming from material absorption at all — it's coming from
the **inversion** (next section).

---

## 4. The `r → n̂` inversion: where artifacts are born

The Fresnel reflection is a **Möbius map** of the index. For s-pol at incidence angle θ in a
medium of index `n₁` onto `n̂`:
```
r = (n₁cosθ − √(n̂² − n₁²sin²θ)) / (n₁cosθ + √(n̂² − n₁²sin²θ))
```
Inverting `r → n̂` is fine where the map is gently sloped, but it has **two dangerous regimes**:

- **Near |r| → 1** (a good reflector — a conductor like CNT, |r| ≈ 0.6–0.93 in our data). Here
  the map compresses a huge range of `n̂` into a thin shell near the unit circle, so the inverse
  *expands* tiny errors in `r` into large swings in `n̂`. The inversion is **ill-conditioned**.
- **Branch / phase-crossing points.** At particular phases of `r` the inverse `n̂(r)` passes a
  near-singular point: `n` shoots up while `k` is squeezed toward zero. **That is the n-spike /
  k-notch you saw** — a property of the *map*, not the material.

So a perfectly smooth material, viewed through an inversion sitting on top of a near-singular
phase, produces a sharp fake peak. The peak's *frequency* is wherever `r(ω)` happens to cross
the bad phase — which is set by **whatever rotates the phase of r with frequency**. Enter the gap.

---

## 5. The gap: Fabry–Pérot fringes and roughness

A contact gap makes SiO₂ | air `d` | CNT an **etalon**:
```
r_meas = (r₁ + r₂ e^{−i2β}) / (1 + r₁ r₂ e^{−i2β}),     2β = (ω/c)·2d·cosθ_gap
```

- **The `e^{−i2β}` term rotates the phase of r linearly with ω**, at a rate set by `d`. Over a
  wide band this makes **Fabry–Pérot fringes** — oscillations periodic in ω with spacing
  `Δω ≈ πc/(d cosθ)`. For a *small* `d` the first fringe is far out of band, so instead of
  visible ripple you get a single gentle phase ramp across the band — which is exactly enough to
  walk `r` into the inversion singularity at some frequency.
- **The fringe positions scale with `d`.** Double the gap, halve the fringe spacing. *Anything
  built on this phase ramp — including an inversion-singularity spike — moves when you change `d`.*

**Roughness (a distribution of gaps).** A rough surface is a spread of `d` values across the spot.
Averaging many etalons washes out the high-frequency fringes — a **Debye–Waller** magnitude
roll-off `W = exp(−2((ω/c)cosθ_gap·σ_d)²)`. Crucially this is a **smooth, broadband magnitude
suppression**: it can lower high-frequency contrast, but it **cannot create or move a narrow
peak**. That asymmetry is a tool: if a feature can be flattened by a broadband-smooth `σ_d`, it
was broadband; if a narrow peak survives every physical `σ_d`, the gap roughness isn't making it.

---

## 6. Windowing and timing: the other two knobs

- **Time-domain window (Hann/Tukey).** Multiplying the trace by a window = **convolving** the
  spectrum with the window's Fourier transform. Effects: real sharp features get **broadened**
  (a too-narrow window smears a true resonance); hard edges add **ringing / side-lobes** (periodic
  ripple — Gibbs). So window width changes *apparent* linewidths and can add ripple — but a real
  `ω₀` doesn't move, it just blurs. Our fixed-width identical window keeps this effect *identical*
  across sample and reference so it cancels in the ratio.
- **Timing offset (sub-sample t₀).** A pure time shift is a **linear phase ramp** `e^{−iωΔt}` —
  algebraically the *same shape* as a small gap. It moves with whatever sets `t₀`. This is why
  self-referencing matters: forming `W = Y₂/Y₁` within a trace cancels the absolute time origin,
  leaving only the *physical* inter-pulse delay (which is pinned by the window thickness). The
  feature that's pinned (real delay) survives; the one that's a labelling choice (t₀) cancels.

---

## 7. The master diagnostic: turn the knob

Everything above collapses to one procedure:

> **Vary a measurement knob (gap `d`, window width, alignment, t₀) and watch the feature.**
> - **Pinned in frequency ⇒ real material** (Drude at 0; Lorentz/phonon at ω₀). Its scale is
>   internal — the knob can't reach it.
> - **Moves / scales / vanishes ⇒ measurement artifact** (FP fringes, inversion singularity,
>   ringing, phase ramps). Its "frequency" is set by the knob, so the knob moves it.

**Our CNT case, start to finish:**
1. Expectation: metallic CNT ⇒ **Drude** ⇒ σ peak at 0, smooth falling, `k ≫ n`. No finite-ω peak.
2. Observation (naive inversion): a sharp peak in `n` at ~0.6 THz — and `k` **dips** there. The
   k-dip already says "not absorption."
3. Knob test: sweep the de-embed gap `d`. The peak slid 0.32 → 0.61 → 0.94 THz for d = −15 → 0 →
   +5 µm. **It moved with the knob ⇒ artifact**, not a resonance.
4. Mechanism: `r` sits near |r|=1; the residual ~13 µm / ~64 fs gap is a linear phase ramp that
   walks `r` through the inversion singularity at ~0.6 THz.
5. Cure: de-embed the gap phase. At the physical `d` the spike **vanishes** and `n, k` become
   smooth and **Drude-like** (`k ≈ 5–6 ≫ n`, broad σ ≈ 400 S/m) — the expected free-carrier shape.
6. Corroboration: a Drude+Lorentz fit barely improved RMS (+4–6%), the two gap models disagreed
   on ω₀ (0.6 vs 1.2 THz), and one oscillator railed to a razor spike — a real resonance would
   have been found consistently and immovably by both.

---

## 8. Cheat-sheet

| Feature in n/k/σ | Origin | Where it's pinned | Confirm by |
|---|---|---|---|
| σ peak at **ω = 0**, smooth falling, k≫n | Drude free carriers | ω=0 (no restoring force) | Doesn't move with any knob; fits bare Drude |
| **k peak** + dispersive n wiggle | Lorentz resonance (phonon/exciton/…) | ω₀ (restoring force) | ω₀ fixed under gap/window/t₀ changes |
| **n spike at a k-DIP** | Inversion singularity (|r|→1) | wherever r hits the bad phase | **Moves with gap d / t₀**; vanishes on de-embed |
| Periodic ripple in ω | Fabry–Pérot fringes | spacing ∝ 1/d | Spacing scales with gap; gone when gap removed |
| Broadband high-f magnitude roll-off | Roughness (Debye–Waller) | not a peak — a smooth slope | Lifts smoothly with σ_d; can't make narrow peaks |
| Broadened features / side-lobe ripple | Time-domain window | linewidth ∝ 1/window-width | Changes with window width; cancels in matched ratio |
| Linear phase ramp (n droop, fake delay) | Timing offset t₀ / small gap | set by t₀ or d | Cancels under self-referencing; moves with t₀ |

**One-line takeaway:** ask of every lineshape, *"what internal energy scale would pin this here?"*
If you can name one (a bond, a plasma density, a delay fixed by the window thickness), it's
probably real. If the only thing setting its position is a number you dialled in, it's an artifact —
and the proof is to dial that number and watch it move.
