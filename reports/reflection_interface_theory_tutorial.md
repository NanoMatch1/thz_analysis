# Reflection through different interfaces: why high-index coupling rescues a conductor

A guided tutorial on *why* a highly reflective (conductive) sample is hard to measure in
reflection, and why coupling it through a high-index medium (SiO₂, Si) fixes it. It pairs
with the runnable model `explorations/reflection_theory/drude_interface_model.py` and its
figure `drude_interface_model.png` (panels referenced below as **[row,col]**, 0-indexed).

Convention (matches the codebase): `n̂ = n − ik`, k ≥ 0 absorption, `exp(−iωt)`.
Companion: `reports/lineshape_and_inversion_tutorial.md` (the general signal-vs-artifact view).

---

## 1. The chain you're inverting

```
free carriers → ε(ω) → n̂ = √ε → r(ω ; N₁) Fresnel → measure → invert r → n̂
        Drude        permittivity   complex index    reflection           (closed form)
```

You measure `r` (referenced to a mirror or a bare surface) and want `n̂`. Two links decide
everything: what the **material** does to `r` (Sections 2–3), and how **well-posed** the
backward step `r → n̂` is (Sections 4–6). The incident medium `N₁` — air vs SiO₂ vs Si —
turns out to control the second.

---

## 2. The material: a Drude conductor [0,0] and [0,1]

Free carriers have **no restoring force**, so the only scales are the plasma frequency `ω_p`
(carrier density) and the scattering rate `1/τ`:

```
ε(ω) = ε∞ − ω_p² / (ω² + iω/τ),     σ(ω) = ε0 ω_p² τ / (1 − iωτ)
```

Two consequences you can read off the panels:

- **[0,1] The conductivity peaks at ω = 0** (the "Drude peak"). A metal is **most conductive,
  hence most reflective, at LOW frequency.** This is the crux: the regime where you most want
  data is the regime the material makes hardest.
- **[0,0] n and k are both large at low frequency** and fall with ω. Large `|n̂|` means the
  material is close to a perfect conductor — and a perfect conductor reflects `r = −1`.

So a Drude conductor drives `r → −1` exactly where you need information most. Hold that thought.

---

## 3. The interface: Fresnel reflection and impedance

For s-polarisation, a wave in medium `N₁` at angle θ hitting a sample `N₂`:

```
r_s = (N₁cosθ − √(N₂² − N₁²sin²θ)) / (N₁cosθ + √(N₂² − N₁²sin²θ))
```

The intuition that makes this *feel* natural is **impedance**. Each medium has a wave
impedance `Z ∝ 1/N` (high index → low impedance). Reflection is an impedance mismatch:

```
r ≈ (Z_sample − Z_incident) / (Z_sample + Z_incident)
```

A conductor has **very low impedance** (`Z ∝ 1/N₂`, and `N₂` is huge). Seen from **air**
(high impedance), the mismatch is enormous → `r ≈ −1`. That's why metals are mirrors.

**Now the key move.** Raise the incident index `N₁` (couple through SiO₂ or Si). That *lowers*
the incident impedance, moving it **closer to the conductor's low impedance** → the mismatch
shrinks → **`r` moves away from −1.** More of the wave couples into the sample, so the
reflection carries more information about `N₂`. This is exactly the principle behind
**ATR / immersion / prism-coupling** spectroscopy of strongly reflecting materials.

**[0,2]** shows it directly: `|r|` for the same Drude conductor drops from ~0.96 (air) →
~0.92 (SiO₂) → ~0.87 (Si) at low frequency. Lower `|r|` = more contrast = more signal in the
quantity you invert.

---

## 4. The inversion and its singularity

The closed form (the *same* formula for every geometry — only `N₁` changes; there is no
separate "open form") inverts `r → n̂`:

```
q = N₁cosθ (1 − r)/(1 + r),     n̂ = √(q² + (N₁sinθ)²)
```

Look at the denominator: **`q → ∞` as `r → −1`.** A perfect mirror (`r = −1`) maps to
`n̂ = ∞`, which is *physically correct* (a perfect conductor has infinite index) but
numerically catastrophic — the map has a **pole at r = −1**. Anything that nudges the measured
`r` near −1 (a conductive sample; or worse, a coupling artifact pushing `|r| > 1`) makes the
inverted `n` blow up. (This is the bug we found on the gold-referenced CNT: n → ∞ ~every 1 THz
wherever the noisy `r` brushed past −1.)

**[1,0]** is the geometric picture: `r(ω)` plotted in the complex plane, zoomed on the
`r = −1` cross. The **air** trajectory hugs the singularity; **SiO₂** and **Si** stand off from
it. The dots mark the 0.1 THz (low-frequency) end — exactly where air is closest to the pole.

---

## 5. Conditioning: distance to the pole = recoverability [1,1]

How ill-posed the inversion is at a given frequency is set by **how close `r` is to −1**,
i.e. **`|1 + r|`**. Near the pole, a small error `δr` in the measured reflection produces a
large error in `n` (roughly `δn ∝ δr / |1+r|²`).

**[1,1]** plots `|1+r|` (log scale). Higher-index incidence lifts it everywhere, most at low
frequency. Concretely, at 0.3 THz for this conductor:

| incidence | \|r\| | **\|1+r\|** (distance to pole) |
|---|---|---|
| air | 0.958 | 0.060 |
| SiO₂ | 0.921 | 0.116 |
| Si | 0.867 | **0.195** |

Si sits **~3× further** from the singularity than air — so a given measurement error does
~3× (in amplitude; ~10× in variance) less damage to the extracted `n`.

---

## 6. The payoff: same noise, very different answer [1,2]

Panel **[1,2]** is the punchline. Take the *same* additive measurement noise on `r`, push it
through the inversion 300 times, and report the **fractional uncertainty in `n`** versus
frequency for each incident medium:

- **Air**: uncertainty explodes toward low frequency — `n` is essentially unrecoverable there
  (this is your "can't access low-f because it's too reflective" problem, quantified).
- **SiO₂**: better by roughly half an order of magnitude.
- **Si**: best — the low-frequency window stays open.

Same sample, same instrument noise. The **only** difference is the impedance of the medium you
looked through. That is the entire case for the window geometry.

---

## 7. The catch: an air gap throws the advantage away

The benefit comes from the wave being **incident from the high-index medium onto the sample**.
That requires **optical contact** — the conductor pressed against (or grown on) the Si/SiO₂.

If a thin **air gap** intrudes, the wave crosses the gap *as air*, so the reflection that
actually carries the sample happens at an **air/CNT** interface again — you snap straight back
to the air-incidence (singular) case, **and** you add the contaminating opposite-sign
SiO₂→air reflection (see the air-gap notes). The gap is doubly destructive: it loses the
impedance advantage *and* injects an interference artifact. **Optical contact is the whole
ballgame.**

---

## 8. Practical reading for the rig

- **Bare reflection (air incidence)** is simple and contact-free, but **ill-conditioned** for a
  conductor — the low-frequency index is buried under noise amplification. Good alignment fixes
  the *wavefront* problem but **not** this conditioning problem.
- **High-index coupling (Si > SiO₂)** reopens the low-frequency window — but needs **gap-free
  contact**.
- The sweet spot for a flat-substrate jig: a **high-resistivity Si** base, measuring reflection
  *through* the Si onto the sample/Si interface, with the conductor pressed or deposited
  directly on the Si (best shot at no gap). You then self-reference to the Si back-face — no
  gold needed.

**Play with it:** open `drude_interface_model.py` and turn the knobs — `PLASMA_FREQUENCY_THZ`
(how metallic), `SCATTERING_TIME_FS` (damping), `INCIDENT_MEDIA`, `NOISE_AMPLITUDE`. Watch the
`r(ω)` trajectory in [1,0] crawl toward or away from the −1 pole, and the uncertainty in [1,2]
rise or fall. Building that reflex — "where is `r` relative to −1, and what medium moves it?" —
is the intuition to keep.
