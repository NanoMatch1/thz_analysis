# Literature canvas — extracting CNT optical constants through a rough contact gap

Date: 2026-06-22. Context: window-coupled THz reflection of a CNT mat pressed on the back
of a SiO2 window. The contact is imperfect AND the CNT surface is rough, so the "air gap"
is really a *distribution* of gaps across the beam spot. n droops below 1; a single-gap
Fabry-Pérot de-embed lifts it but is degenerate in the gap thickness d (see
ANALYSIS_NOTES §9/§9b, `air_gap_deembedding` memory).

## The reframing (why "a combination of gaps" is the right physics)

- A *single* gap → one Fabry-Pérot etalon (what we de-embed): adds a delay (linear phase)
  + multiple-bounce ripple.
- A *rough* contact → the beam spot averages over many gap thicknesses → the measured
  reflection is the spot-average of many etalons. Two effects (both already in our
  `explore_air_gap_deembedding.py` roughness demo): (1) the *mean* gap still adds a delay;
  (2) the *spread* σ_d damps the high-frequency recovery — a coherence-bandwidth / Debye–
  Waller ceiling. This is exactly the rough-surface specular-reflection problem, which is
  well studied in THz.

## A. Modelling routes (extend the de-embed)

1. **Statistical-gap Fabry-Pérot with a Rayleigh roughness factor.** THz rough-surface
   work corrects the specular reflection with a Rayleigh roughness factor from Kirchhoff /
   Beckmann–Kirchhoff theory — a Debye–Waller-type damping ~exp[−(2σ (ω/c) cosθ)²] — and
   uses the rms roughness together with the THz wave velocity to *correct the extracted
   refractive index*. Applied to our gap: average the etalon over a Gaussian gap-height
   distribution → a *mean* delay d̄ (the linear phase) + a Gaussian coherence factor set by
   σ_d. Fit BOTH (a 2-parameter per-measurement fit). Our demo already does the forward
   average; this is the inverse. Limitation: the coherence ceiling means high-f data is
   intrinsically lost — fit only below it.

2. **Effective-medium / graded-transition layer.** A rough CNT surface pressed on glass is
   a graded air→CNT mixing layer, not a sharp interface. Model it as an effective-medium
   gradient (Bruggeman / Maxwell–Garnett of air + CNT) — the standard THz treatment for
   porous/rough media and graded anti-reflection layers, and already used for CNT films
   (Maxwell–Garnett + Drude–Lorentz). Replaces "vacuum gap + sharp interface" with "a
   few-µm effective layer with a fill-fraction profile" — more physical for a fluffy mat.

3. **Minimum-phase / pulse-delay d as seed, then a PRESSURE-SERIES global fit.** Anchor d
   with the independently-measured pulse round-trip delay, cross-check with minimum-phase,
   then use a pressure series (we have a pressure holder) to break the degeneracy: shared
   material r_back(ω), per-pressure (d̄, σ_d); extrapolate to high pressure (zero gap). Most
   robust computational path, uses existing hardware.

## B. Experimental routes (often the cleaner fix)

1. **Transmission geometry — the CNT field-standard.** Most CNT THz conductivity work is
   done in *transmission* (free-standing film, or on a thin known substrate), fit with
   Drude–Smith / Drude–Lorentz and Maxwell–Garnett EMT. No contact gap at all. A
   transmission measurement of the same CNT material would sidestep the problem and give a
   ground-truth cross-check of the reflection numbers.
2. **Index-matching layer.** THz index-matching solutions exist (e.g. TeraSol: BaTiO3 +
   benzocyclobutene, tunable n = 1.8–5). A thin matching layer (≤ λ/10) between sample and
   window kills the gap reflection. Caveat: infiltrating a CNT mat may alter it — but a thin
   matching film on the window side may be tolerable.
3. **Pressure series / higher pressure.** Controlled pressure (up to diamond-anvil-cell
   THz, or the ATR pressure-application devices) reduces and characterises the gap;
   extrapolate to contact.
4. **Direct deposition / growth on the window.** If CNTs can be deposited directly on the
   SiO2 back face, there is no gap — the ultimate fix.
5. **ATR with controlled pressure.** Evanescent-field geometry, very interface-sensitive;
   pressure devices exist to ensure contact.
6. **Tinkham sheet-conductance framing.** If the mat is electrically thin, extract the
   sheet conductance (more gap-robust than n,k) via the (modified) Tinkham thin-film
   formula, rather than the full n.

## Recommendation (what's worth trying)

- **No new hardware:** (a) the 2-parameter statistical-gap fit (d̄, σ_d), with d̄ anchored by
  the pulse round-trip delay — directly models the rough gap and explains the high-f
  rolloff; (b) a pressure-series global fit (shared material, per-pressure gap stats).
- **Highest-payoff cross-check:** a transmission measurement of the same CNT material — the
  field standard; validates (or refutes) the reflection extraction.
- **If contact is the killer:** index-matching layer or direct deposition on the window.

## Sources

- Lai et al. 2013, novel THz reflection method (window-reflection de-embed): https://journals.sagepub.com/doi/10.1366/12-06713
- Improved sample characterization in THz reflection imaging/spectroscopy: https://pubmed.ncbi.nlm.nih.gov/19259226/
- Robust phase correction for THz reflection TDS (arXiv): https://arxiv.org/pdf/2412.18662
- Methods for material properties with THz-TDS, optically thick (arXiv): https://arxiv.org/pdf/2403.18531
- Modified Beckmann–Kirchhoff for slightly rough surfaces at THz (IEEE): https://ieeexplore.ieee.org/document/8889095
- Roughness parameters of metallic surfaces from THz reflection (Opt. Lett.): https://opg.optica.org/ol/abstract.cfm?uri=ol-34-13-1927
- Effect of surface roughness on THz reflection measurement: https://www.researching.cn/articles/OJ86636907ec043e82
- THz-TDS in different CNT thin films: https://www.researchgate.net/publication/258712381
- THz conductivity of anisotropic SWCNT films: https://www.researchgate.net/publication/242215055
- DC conductivity of MWCNT films & graphene from noncontact THz: https://www.academia.edu/84443946
- THz charge transport in pristine/doped SWCNT films (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S0008622317310722
- Effective medium theories in the THz regime (IntechOpen): https://www.intechopen.com/chapters/6533
- Ultra-broadband THz graded-index AR coating (polymer composites): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6418670/
- ATR for THz — review (MDPI Appl. Sci.): https://www.mdpi.com/2076-3417/10/14/4688
- THz ATR method with pressure-application device (patent): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11774357
- Modified Tinkham equation for thin-film THz complex conductivity (Springer): https://link.springer.com/article/10.1007/s10762-023-00928-z
- Sheet conductance & imaging of graphene by THz-TDS: https://www.researchgate.net/publication/312566519
- Improved model to extract 2D-material conductivity from THz transmission/reflection (MDPI): https://www.mdpi.com/2079-9292/12/4/864
- THz refractive-index matching solution (TeraSol): https://www.researchgate.net/publication/332879297
- THz-TDS under high pressure in a diamond anvil cell (Rev. Sci. Instrum.): https://pubs.aip.org/aip/rsi/article/97/1/013902/3376054
- THz info-acquiring apparatus, air-gap handling (patent): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9316582
