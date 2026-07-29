## Literature

This exact trick — using a reflection from a fixed reference surface within the *same* trace as an internal clock, so that scan-to-scan drift cancels algebraically — has a well-established pedigree in THz-TDS reflection work:

1. **Pashkin, Kempa, Němec, Kadlec & Kužel**, *"Phase-sensitive time-domain terahertz reflection spectroscopy,"* Rev. Sci. Instrum. **74**, 4711 (2003). The foundational paper on getting an accurate, reproducible phase out of a THz reflection measurement by referencing to a fixed reflection plane — the direct ancestor of the front/back-window trick you're using.

2. **Nashima, Morikawa, Takata & Hangyo**, *"Measurement of optical properties of highly doped silicon by terahertz time domain reflection spectroscopy,"* Appl. Phys. Lett. **79**, 3923 (2001). Same family — reports relative-phase accuracy below 10 mrad at 1 THz using a reference-reflection geometry.

3. **Gorecki, Klokkou, Piper, Mailis, Papasimakis & Apostolopoulos**, *"High-precision THz-TDS via self-referenced transmission echo method,"* Appl. Opt. **59**, 6744 (2020). The closest direct analogue of your algebra: they divide the directly-transmitted pulse by its own internal-reflection **echo**, within one trace, specifically to cancel laser drift and mechanical jitter — same cancellation, transmission-echo geometry instead of front/back reflection.

4. **Lordon, Giraldo Betancur & Gallot**, *"Self-referenced terahertz time-domain ATR spectroscopy of solutions,"* Appl. Phys. Lett. **128**, 221107 (2026). Near-identical experimental geometry to yours — ATR off a window, self-referenced specifically to remove alignment/jitter error between sample and reference scans.

(Your pipeline already cites arXiv:2412.18662 for the *phase-correction* step downstream of this — that's a different, complementary problem: fixing residual misplacement phase, not the front-pulse jitter-cancellation shown here.)

## The equation

The whole mechanism rests on one fact: **a pulse arriving later just means its phase spins faster with frequency.** A pulse that arrives at time *t* has, at frequency *f*,

$$\varphi(f) = -2\pi f\, t$$

Every trace records two pulses: a **first** reflection (front of the window, arrives at $t_1$) and a **second** reflection (back of the window — the sample, arrives at $t_2$). Dividing two spectra is the same as *subtracting* their arrival times:

$$\underbrace{\varphi_2 - \varphi_1}_{\text{divide 2nd by 1st pulse}} = -2\pi f\,(t_2 - t_1) = -2\pi f\,\Delta t$$

Apply that within each trace:

$$\text{reference: } -2\pi f\,\Delta t_{\text{ref}} \qquad\qquad \text{sample: } -2\pi f\,\Delta t_{\text{sample}}$$

Notice $t_{1,\text{ref}}$ and $t_{1,\text{sample}}$ — the absolute, arbitrary, drifting arrival times — are **already gone**. Each trace only remembers its own internal spacing, not when it happened to start.

Now compare sample to reference — the actual measurement:

$$\;\Delta\varphi_{\text{sample}}(f) = -2\pi f\,\big(\Delta t_{\text{sample}} - \Delta t_{\text{ref}}\big)\;$$

That's it. The absolute start times never appear in this final line — not as a residual, not as an error term — because they cancelled two steps earlier. All that survives is the difference between the two round-trip delays, which is exactly the sample's signature.

Sources:
- [Self-referenced terahertz time-domain ATR spectroscopy of solutions | AIP Publishing](https://pubs.aip.org/aip/apl/article-abstract/128/22/221107/3393805/Self-referenced-terahertz-time-domain-ATR)
- [High-precision THz-TDS via self-referenced transmission echo method | Optica](https://opg.optica.org/ao/abstract.cfm?uri=ao-59-22-6744)
- [Measurement of optical properties of highly doped silicon by terahertz time domain reflection spectroscopy | AIP Publishing](https://pubs.aip.org/aip/apl/article-abstract/79/24/3923/516269/Measurement-of-optical-properties-of-highly-doped)
- [Robust phase correction techniques for terahertz time-domain reflection spectroscopy (arXiv:2412.18662)](https://arxiv.org/pdf/2412.18662)