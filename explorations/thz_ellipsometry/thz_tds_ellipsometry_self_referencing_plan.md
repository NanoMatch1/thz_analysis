# Self-Referenced Reflection THz-TDS by Polarization Switching

**Theory, measurement plan, and inversion to n, k and ε for isotropic and anisotropic bulk samples**

---

## 1. Purpose and scope

This document describes a reflection THz-TDS method that needs no reference mirror. The sample stays in its measurement position and no optic in the THz path moves. The reference is the sample itself, measured in orthogonal THz polarizations. The input polarization is switched at the source by rotating the magnet of the spintronic emitter. For anisotropic samples, the detected polarization is selected by rotating the probe polarization in the GaP electro-optic (EO) crystal.

There are two operating modes:

- **Isotropic mode (fast).** Switch the input between $s$ and $p$ with a single, fixed probe setting. Two measurements per sample give the complex permittivity $\varepsilon$ and refractive index $\tilde n$. This mode is for samples known to be isotropic, such as silicon or the calibration standards.
- **GE mode (probe-polarization generalized ellipsometry).** Use two input polarizations (magnet) and two probe polarizations. Four measurements give the full normalized reflection Jones matrix. This mode is for samples that are uniaxial with the optic axis in the surface plane, and it also serves as a diagnostic for unexpected anisotropy.

Out-of-plane anisotropy (optic axis along the surface normal) cannot be resolved by reflection at a single angle, whatever polarization scheme is used. It is handled with transmission or a variable angle of incidence (§7.9). Thin films are measured in transmission.

---

## 2. Why mirror referencing fails

Conventional reflection THz-TDS divides the sample spectrum by the spectrum of a reference mirror placed in the same plane. Two problems make this unreliable for real samples.

**Phase from displacement.** A displacement $d$ of the reflecting plane along its normal changes the optical path by $2d\cos\theta$. At $\theta = 45°$ and 1 THz ($\lambda$ = 300 µm), a 1 µm placement error gives about 1.7° of phase error, and the error grows linearly with frequency.

**Changed beam geometry.** Tilt, non-planarity and roughness change where the reflected beam lands on the GaP and at what angle. This changes the frequency-dependent overlap between the THz field and the probe. A mirror and a sample therefore have different geometric transfer functions, and dividing by the mirror spectrum leaves a geometry error in the result rather than cancelling the system response.

---

## 3. Principle: ellipsometric self-referencing

### 3.1 Signal model (isotropic sample)

Let $S_p(\omega)$ and $S_s(\omega)$ be the detected spectra for $p$- and $s$-polarized incidence. Each is a product of factors:

$$S_p(\omega) = A(\omega)\,C_p(\omega)\,a\;r_p(\omega)\;G(\omega), \qquad S_s(\omega) = A(\omega)\,C_s(\omega)\,b\;r_s(\omega)\;G(\omega)$$

| Symbol | Meaning |
|---|---|
| $A(\omega)$ | Emitter spectrum |
| $C_p, C_s$ | Polarization-dependent system factors: emitter anisotropy between magnet states, OAP polarization effects |
| $a, b$ | EO detection coefficients for $p$- and $s$-polarized THz at the GaP (real constants, §3.2) |
| $r_p, r_s$ | Fresnel reflection coefficients of the sample |
| $G(\omega)$ | Everything geometric: sample height (phase), tilt, beam walk, THz/probe overlap at the GaP, specular loss from roughness |

Both polarizations follow exactly the same rays off exactly the same surface, so $G$ is identical for the two and cancels in the ratio:

$$R(\omega) \equiv \frac{S_p(\omega)}{S_s(\omega)} = \underbrace{\frac{a\,C_p(\omega)}{b\,C_s(\omega)}}_{\kappa(\omega)}\;\rho(\omega), \qquad \rho(\omega) \equiv \frac{r_p(\omega)}{r_s(\omega)}$$

The system factor $\kappa(\omega)$ is measured once on a known reflector (§5). That calibration does not depend on where the reflector sits, because placement errors multiply $S_p$ and $S_s$ by the same factor, which divides out.

### 3.2 The EO crystal as a polarization analyzer

For a fixed GaP azimuth and probe polarization, the EO signal is linear in the THz field at the crystal: $S = a\,E_p + b\,E_s$. GaP is cubic, so its linear optical properties are isotropic. A $p$- or $s$-polarized THz pulse sees the same THz refractive index, the same velocity matching with the probe, and the same $r_{41}$ dispersion. Only the geometric projection through the $\chi^{(2)}$ tensor depends on polarization. The coefficients $a$ and $b$ are therefore real and independent of frequency, and they are set entirely by the crystal axes and the probe polarization (§7.3).

This has two consequences. In isotropic mode, no analyzer is needed before the detector, because $a/b$ is simply absorbed into $\kappa$. In GE mode, changing the probe polarization changes $(a, b)$, so the GaP itself provides the two detection projections without any element in the THz beam.

### 3.3 Roughness

At THz wavelengths, many surfaces that look rough by eye behave as specular reflectors. In the scalar (Kirchhoff) approximation, roughness with RMS height $\sigma$ reduces the specular intensity by $\exp[-(4\pi\sigma\cos\theta/\lambda)^2]$. For $\sigma$ = 10 µm at 1 THz and 45°, about 92% of the specular power remains. To first order this loss is the same for both polarizations, so it sits inside $G$ and cancels. That stops being true once $\sigma$ approaches about $\lambda/10$.

### 3.4 The angle of incidence is set by the optics, not the sample

The focused THz beam is a cone of plane-wave components, and the collection optics accept a cone of outgoing directions. For specular reflection, the angle of incidence of any in/out ray pair is half the angle between the incident and outgoing ray directions. The sample's orientation only decides which pairs satisfy the law of reflection.

Measure angles relative to the design axes. If the sample is tilted in-plane by $\Delta$, an incident component at angle $\alpha$ leaves at $\beta = \alpha - 2\Delta$, and its angle of incidence is $\theta = \theta_0 + (\alpha + \beta)/2 = \theta_0 + \alpha - \Delta$. The detected rays must lie in both the incident cone and the acceptance cone, so their centroid is near $\alpha \approx \Delta$, which gives $\theta_{\text{eff}} \approx \theta_0$. For Gaussian input and acceptance angular profiles with widths $\sigma_{\text{in}}$ and $\sigma_{\text{out}}$, the remaining shift is

$$\theta_{\text{eff}} - \theta_0 \approx \Delta\,\frac{\sigma_{\text{in}}^2 - \sigma_{\text{out}}^2}{\sigma_{\text{in}}^2 + \sigma_{\text{out}}^2}.$$

This only holds while the collection optics stay fixed. Re-steering the optics after the sample to follow a tilted beam genuinely changes the angle, to $\theta_0 - \Delta$.

---

## 4. Plan: switching the THz polarization at the source

### 4.1 Spintronic emitter and magnet rotation

In the spintronic emitter, the inverse spin Hall effect produces a THz field polarized perpendicular to the in-plane magnetization $\mathbf{M}$. Rotating the magnet by an angle $\beta_M$ rotates the THz polarization by the same angle. A 90° rotation switches between $p$ and $s$ incidence while every optic stays fixed. A 180° rotation inverts the THz field, which is used to remove background.

The magnet must saturate the ferromagnetic layer in every state so that emission is reproducible. The emitter substrate must not be birefringent at THz frequencies; fused silica, glass and c-cut sapphire at normal incidence are suitable. Any remaining reproducible difference between magnet states is absorbed by the calibration.

Because the emission is linear in the direction of $\mathbf{M}$, a continuous magnet sweep on a gold reflector provides a known, rotatable input polarization. This is used to calibrate the detection in GE mode (§7.5).

### 4.2 Acquisition sequence

Record time traces at magnet angles 0°, 90°, 180° and 270°, where 0° gives $p$ incidence. Cycle through the states repeatedly during the measurement rather than recording each one in a single block, so that slow drift is spread equally over all states. Average the traces for each state over all cycles before forming any ratio. A ratio of averages is less biased at low signal-to-noise than an average of ratios.

Non-magnetic background (optical rectification in the substrate, pump leakage, electronic pickup, static GaP birefringence) does not reverse with $\mathbf{M}$, so it is removed by taking differences:

$$S_p(t) = \tfrac{1}{2}\left[S(t;\,0°) - S(t;\,180°)\right], \qquad S_s(t) = \tfrac{1}{2}\left[S(t;\,90°) - S(t;\,270°)\right]$$

In GE mode, the same magnet sequence is repeated at each probe setting. Magnet states cycle quickly within a probe setting, and the probe settings alternate in repeated blocks.

### 4.3 Detector configuration

The GaP is mounted once, with $[\bar{1}10]$ along $p$ and $[001]$ along $s$, and is not touched again. In this orientation, the probe angle alone selects which THz polarization is detected (§7.3). There is no wire-grid analyzer in the THz path.

**Isotropic mode.** Set the probe at $\phi = \tfrac12\arctan 2 \approx 31.7°$ from $[001]$. At this angle the detector is equally sensitive to $E_p$ and $E_s$, at 0.89 of the maximum $s$ sensitivity. One probe setting is used for both magnet states, so no waveplate moves during the measurement.

**GE mode.** Use the two probe settings described in §7.3.

### 4.4 Setting the magnet zero (wire grid used for alignment only)

If the emitted polarization is off by an angle $\delta$ from true $p$, the "$p$" measurement picks up some $s$ signal: $S_p \propto a\,r_p\cos\delta + b\,r_s\sin\delta$. This leakage does not cancel in the calibration, because $r_s/r_p$ differs between the calibration standard and the sample. A misalignment of 1° gives an error of a few percent.

To set $\delta$ accurately, temporarily insert the wire grid with its transmission axis along $s$ and rotate the magnet until the THz signal nulls; that angle emits pure $p$. Rotate the grid by 90° and null again to confirm $s$. Remove the grid afterwards. In GE mode, the magnet-sweep calibration (§7.5) fits any remaining offset as well.

### 4.5 Sample alignment without a visible laser

Rough samples are close to specular at THz wavelengths, so the THz peak amplitude itself can be used as the tilt sensor. Scan the sample tilt about both axes and set each to the centre of the signal peak. Do not re-steer the optics after the sample (§3.4). Out-of-plane tilt is more harmful than in-plane tilt, because it rotates the local $p/s$ basis and mixes the two polarizations.

---

## 5. Calibration with two standards

### 5.1 Gold gives κ(ω)

For a good conductor, $\rho \approx -1$ at any angle in the convention of §6.1; the deviation for gold at 1 THz is about $10^{-3}$. Therefore

$$\kappa(\omega) = \frac{R_{\text{Au}}(\omega)}{\rho_{\text{Au,model}}(\omega)} \approx -R_{\text{Au}}(\omega).$$

κ depends on the probe setting, so measure it at the probe setting used for isotropic mode. A Drude model of gold can be used for $\rho_{\text{Au,model}}$ if the $10^{-3}$-level deviation matters.

### 5.2 High-resistivity silicon gives θ_eff

High-resistivity silicon is lossless and essentially dispersion-free below about 3 THz, with $n = 3.4175$. After dividing its measured ratio by κ, fit θ as the single free parameter:

$$\theta_{\text{eff}} = \arg\min_\theta \sum_\omega \left|\frac{R_{\text{Si}}(\omega)}{\kappa(\omega)} - \rho_{\text{Si}}(\theta)\right|^2$$

At exactly 45°, $\rho_{\text{Si}} = -0.651$, constant across the band. A built-in check is that the phase of the κ-corrected silicon data should be flat at π. Remove the silicon back-face echo by time-windowing (it arrives about 11 ps after the main pulse for a 0.5 mm wafer) or use a thick window. $\theta_{\text{eff}}$ does not depend on the probe setting and is shared by both modes.

### 5.3 Testing that the angle stays pinned

Tilt the silicon by ±1–2° in-plane and refit $\theta_{\text{eff}}$. If it doesn't change, the input and collection cones are matched and sample mounting tilt does not affect the angle. If it does change, the slope measures the asymmetry factor of §3.4 and gives the angle uncertainty for a given mounting error.

---

## 6. Isotropic mode: from measured traces to n, k and ε

### 6.1 Conventions

The Fresnel coefficients use the ellipsometric (Azzam–Bashara / Born & Wolf) sign convention, in which $r_p = -r_s$ at normal incidence. For ambient air, sample permittivity ε, and angle of incidence θ:

$$N \equiv \sqrt{\varepsilon - \sin^2\theta}\;\;(= \tilde n\cos\theta_t), \qquad r_s = \frac{\cos\theta - N}{\cos\theta + N}, \qquad r_p = \frac{\varepsilon\cos\theta - N}{\varepsilon\cos\theta + N}$$

Use the same convention in the gold model, the silicon model, the inversion and the anisotropic forward models of §7. The sign and handedness of the hardware are absorbed by the calibration.

**Fourier convention.** Physics usually writes fields as $e^{-i\omega t}$, which gives $\tilde n = n + ik$ and $\operatorname{Im}\varepsilon > 0$ for absorbing media. NumPy's forward `fft` produces spectra that correspond to $e^{+i\omega t}$, where absorption appears with the opposite sign. Either take the complex conjugate of measured ratios before inversion, or reverse the branch conditions. Choose one approach and use it everywhere.

### 6.2 Time traces to spectra

Apply the same time window to $S_p(t)$ and $S_s(t)$. Make it wide enough to contain both reflected pulses fully (their shapes differ because $r_p \neq r_s$) and short enough to exclude echoes from the emitter, the GaP and the sample's back face. Zero-pad if a finer frequency grid is wanted, then Fourier transform. Keep only frequencies where both $|S_p|$ and $|S_s|$ are about ten times above the noise floor or more.

### 6.3 Ratio and calibration

$$R_{\text{sample}}(\omega) = \frac{S_p(\omega)}{S_s(\omega)}, \qquad \rho(\omega) = \frac{R_{\text{sample}}(\omega)}{\kappa(\omega)}$$

### 6.4 Inversion for a bulk isotropic sample

Using Snell's law ($\sin\theta = \tilde n\sin\theta_t$), the Fresnel ratio can be written compactly:

$$\rho = \frac{r_p}{r_s} = -\frac{\cos(\theta + \theta_t)}{\cos(\theta - \theta_t)}$$

Form $Q \equiv (1-\rho)/(1+\rho)$ and expand the cosines:

$$Q = \frac{\cos(\theta-\theta_t) + \cos(\theta+\theta_t)}{\cos(\theta-\theta_t) - \cos(\theta+\theta_t)} = \frac{\cos\theta\,\cos\theta_t}{\sin\theta\,\sin\theta_t}$$

Substituting $\sin\theta_t = \sin\theta/\tilde n$ and $\cos\theta_t = N/\tilde n$ gives a result that is linear in $N$:

$$N = \frac{\sin^2\theta}{\cos\theta}\,Q$$

Since $N^2 = \varepsilon - \sin^2\theta$:

$$\boxed{\;\varepsilon(\omega) = \sin^2\theta + N^2 = \sin^2\theta\left[1 + \tan^2\theta\left(\frac{1-\rho}{1+\rho}\right)^2\right], \qquad \tilde n = n + ik = \sqrt{\varepsilon}\;\;(\operatorname{Re}\tilde n > 0)\;}$$

Use $\theta = \theta_{\text{eff}}$. At exactly 45° the expression simplifies to $\varepsilon = \tfrac12[1 + Q^2]$. One complex value of ρ per frequency determines the two real unknowns $n$ and $k$ at that frequency, with no dispersion model needed.

### 6.5 Built-in checks

**Silicon.** Inverting the silicon data should return $\varepsilon \approx 11.68$, flat across the band. If the sign convention has been mixed up, which effectively replaces $\rho$ with $-\rho$, silicon comes out near 0.52.

**Physical branch.** In the $e^{-i\omega t}$ convention, an absorbing sample must give $\operatorname{Re}N > 0$ and $\operatorname{Im}\varepsilon \ge 0$. Because $N$ follows linearly from $Q$, a violation points to a convention or calibration error, not a square-root branch problem.

**Abelès condition.** At exactly 45°, every isotropic material satisfies $r_p = r_s^2$ in this convention, so $\rho = r_s$.

---

## 7. Anisotropic samples

### 7.1 What a reflection measurement can and cannot determine

At a single angle of incidence and sample azimuth, reflection is fully described by the 2×2 Jones matrix

$$J = \begin{pmatrix} r_{pp} & r_{ps} \\ r_{sp} & r_{ss} \end{pmatrix},$$

where the first index is the output polarization and the second the input. Self-referenced data give $J$ only up to a common complex factor, so at most three complex ratios: $r_{pp}/r_{ss}$, $r_{ps}/r_{ss}$ and $r_{sp}/r_{ss}$.

Two separate questions arise. *Can the full $J$ be measured?* That depends only on the instrument: two independent input polarizations (the magnet) and two independent detection projections (the probe polarization). *Does $J$ contain enough information about the sample?* That depends on the sample's symmetry. If it doesn't, more information has to come from changing the physics of the measurement: the azimuth, the angle of incidence, or a transmission measurement.

| Sample class | Off-diagonal terms? | Complex unknowns | Approach |
|---|---|---|---|
| Isotropic | None | 1 | Isotropic mode |
| Uniaxial, optic axis along the surface normal | None | 2 | Not solvable by reflection at one angle. Use transmission for $\varepsilon_\parallel$ or a variable angle of incidence (§7.2, §7.9) |
| Uniaxial, optic axis in the surface plane | Yes, unless aligned | 2 (+ azimuth) | GE mode at one unaligned azimuth (§7.3–7.6); or isotropic mode at two aligned azimuths (§7.2) |
| Biaxial, or optic axis tilted out of the surface | Yes | 3 or more | GE mode at two or more azimuths plus a model fit; out-of-plane component weakly determined |

Generalized ellipsometry adds no information for samples that do not mix $p$ and $s$. For those, the off-diagonal terms are zero and GE reduces to ordinary ellipsometry.

### 7.2 Reflection from an anisotropic medium with axes aligned to the lab frame

Use lab axes $x$ (in the surface, within the plane of incidence), $y$ (in the surface, along $s$) and $z$ (the surface normal). If the sample's principal axes coincide with these, $J$ is diagonal, and

$$r_{ss} = \frac{\cos\theta - k_s}{\cos\theta + k_s},\;\; k_s = \sqrt{\varepsilon_{yy} - \sin^2\theta}; \qquad r_{pp} = \frac{\varepsilon_{xx}\cos\theta - k_p}{\varepsilon_{xx}\cos\theta + k_p},\;\; k_p = \sqrt{\frac{\varepsilon_{xx}(\varepsilon_{zz} - \sin^2\theta)}{\varepsilon_{zz}}}$$

The $s$-wave sees only $\varepsilon_{yy}$. The $p$-wave sees $\varepsilon_{xx}$ and $\varepsilon_{zz}$. With all three equal, these reduce to the isotropic Fresnel coefficients of §6.1.

**Optic axis along the surface normal** ($\varepsilon_{xx} = \varepsilon_{yy} = \varepsilon_\parallel$, $\varepsilon_{zz} = \varepsilon_\perp$). One complex ρ cannot determine two complex unknowns. Applying the isotropic inversion returns a pseudo-dielectric function that is not an average of the two components. For example, $\varepsilon_\parallel = 15$, $\varepsilon_\perp = 7$ at 45° gives 16.3. If $\varepsilon_\parallel$ is known independently, for instance from normal-incidence transmission, which probes only in-plane fields, $\varepsilon_\perp$ follows in closed form:

$$r_{pp} = \rho\,r_{ss}(\varepsilon_\parallel), \qquad k_p = \varepsilon_\parallel\cos\theta\,\frac{1 - r_{pp}}{1 + r_{pp}}, \qquad \varepsilon_\perp = \frac{\varepsilon_\parallel\sin^2\theta}{\varepsilon_\parallel - k_p^2}$$

Reflection is less sensitive to $\varepsilon_\perp$ than to $\varepsilon_\parallel$. Errors in ρ grow by roughly a factor of two to three when propagated into $\varepsilon_\perp$.

**Optic axis in the surface plane, at an aligned azimuth.** With the axis along $x$ ($\psi = 0°$): $\varepsilon_{xx} = \varepsilon_e$ and $\varepsilon_{yy} = \varepsilon_{zz} = \varepsilon_o$. With the axis along $y$ ($\psi = 90°$): $\varepsilon_{yy} = \varepsilon_e$ and $\varepsilon_{xx} = \varepsilon_{zz} = \varepsilon_o$. Isotropic-mode measurements at both azimuths give

$$\rho_{0°} = \frac{r_{pp}(\varepsilon_e, \varepsilon_o)}{r_{ss}(\varepsilon_o)}, \qquad \rho_{90°} = \frac{r_{pp}(\varepsilon_o, \varepsilon_o)}{r_{ss}(\varepsilon_e)},$$

which are two complex equations for two complex unknowns. Solve them numerically at each frequency, starting from the isotropic inversions of each ratio. Each azimuth is self-referenced, so changes in $G$ between azimuths cancel, and $\theta_{\text{eff}}$ stays pinned by the optics. Rotation about the sample normal must not wobble, since that introduces out-of-plane tilt. To find the aligned azimuths, cross the wire grid temporarily ($p$ input, grid passing $s$) and rotate the sample to the cross-polarized nulls; the value of ρ at each null identifies which azimuth it is. This route serves as a cross-check of GE mode (§7.7).

### 7.3 Probe-polarization GE: principle

For (110) GaP, let $\phi$ be the probe polarization angle measured from $[001]$. The EO signal produced by the THz field components along the two in-plane crystal axes is (Planken et al., 2001)

$$S \;\propto\; E_{[001]}\sin 2\phi \;+\; 2\,E_{[\bar 1 10]}\cos 2\phi.$$

Physically, a THz field along $[\bar 1 10]$ induces birefringence with eigenaxes at ±45° to the crystal axes, while a field along $[001]$ induces half as much birefringence with eigenaxes along the crystal axes. The detected signal is largest when the probe lies at 45° to the induced eigenaxes.

With the crystal mounted as in §4.3 ($[\bar 1 10]$ along $p$, $[001]$ along $s$):

| Probe setting | Detects | Relative weight |
|---|---|---|
| $\phi = 0°$ or $90°$ (probe along $s$ or $p$) | $E_p$ only | 2 |
| $\phi = 45°$ | $E_s$ only | 1 |
| $\phi \approx 31.7°$ (isotropic mode) | $E_p$ and $E_s$ equally | 0.89 each |

A 45° rotation of the probe polarization therefore switches cleanly between the two orthogonal THz components. Nothing in the THz beam moves, and the projection comes from tensor symmetry rather than a wire grid. This means no extinction-ratio limit (crystal strain and probe polarization purity play that role instead) and no frequency dependence in the projection. This is the polarization-resolved EO scheme of van der Valk, van der Marel and Planken (2005).

### 7.4 Measurement model and Jones-matrix extraction

Label the probe settings $k \in \{1, 2\}$ and the input polarizations $j \in \{p, s\}$. After background subtraction and Fourier transform, the four signals form a 2×2 matrix

$$\mathbf{S}(\omega) = c(\omega)\;\mathbf{D}\;J(\omega)\;\mathbf{E}(\omega), \qquad \mathbf{E} = \operatorname{diag}(E_p, E_s)$$

where $c(\omega)$ is the common factor (emitter spectrum, geometry $G$, overall EO response), $\mathbf{D}$ is the 2×2 detection matrix with rows $(a_k, b_k)$, and $\mathbf{E}$ holds the input-side system factors. Ideally $\mathbf{D} \propto \begin{pmatrix} 2 & 0 \\ 0 & \pm 1 \end{pmatrix}$; small angle offsets add off-diagonal entries. With $\mathbf{D}$ calibrated (§7.5), compute

$$\mathbf{M} = \mathbf{D}^{-1}\mathbf{S} = c\,J\,\mathbf{E}, \qquad M_{ij} = c\,r_{ij}\,E_j.$$

The common factor $c$ (including $G$) drops out of the normalized Jones matrix:

$$\frac{r_{pp}}{r_{ss}} = \frac{M_{pp}}{M_{ss}}\cdot\frac{E_s}{E_p}, \qquad \frac{r_{ps}}{r_{ss}} = \frac{M_{ps}}{M_{ss}}, \qquad \frac{r_{sp}}{r_{ss}} = \frac{M_{sp}}{M_{ss}}\cdot\frac{E_s}{E_p}$$

This requires the input-side ratio $E_p/E_s$ separately from $\mathbf{D}$, not only their product. Gold alone gives only the product, which is why the detection matrix is pinned by EO theory plus a magnet sweep (§7.5). Once $\mathbf{D}$ is known, gold gives $E_p/E_s = (M_{\text{Au},pp}/M_{\text{Au},ss})/\rho_{\text{Au}}$.

### 7.5 Calibrating the detection matrix with a magnet sweep on gold

From EO theory, $\mathbf{D}$ is known up to a common scale and a few angles: the crystal-axis rotation $\gamma$ relative to the $p/s$ frame, the probe-angle offset $\phi_0$, and, if needed, the waveplate retardance error. With gold in place, sweep the magnet angle $\beta$ through 360° in steps at each probe setting. The input polarization is then a known rotation, $\mathbf{E}_{\text{in}}(\beta) \propto (E_p\cos(\beta - \beta_0),\; E_s\sin(\beta - \beta_0))$, and gold's Jones matrix is diagonal and known. Fitting the sinusoidal dependence on $\beta$ at both probe settings determines $\gamma$, $\phi_0$, $\beta_0$ (common to all frequencies) and the complex ratio $E_p/E_s$ (per frequency). With both probe settings, the data over-determine these unknowns, and adding more probe angles increases the redundancy.

Each probe setting should be normalized by its own probe power, which balanced detection ($\Delta I/I$) does automatically. Otherwise a change in probe power between settings would appear as an error in the relative scale of the rows of $\mathbf{D}$.

### 7.6 Inversion for in-plane uniaxial samples

In the lab frame, a uniaxial sample with its optic axis in the surface at azimuth $\psi$ from the plane of incidence has the permittivity tensor

$$\boldsymbol{\varepsilon}_{\text{lab}} = R_z(\psi)\,\operatorname{diag}(\varepsilon_e, \varepsilon_o, \varepsilon_o)\,R_z(\psi)^{\mathsf T}.$$

For $\psi \neq 0°, 90°$, $J$ has off-diagonal terms and no simple closed form. Calculate $J$ with a 4×4 transfer-matrix forward model (Berreman) for a semi-infinite anisotropic medium:

1. Solve for the two transmitted eigenmodes, keeping the ones that decay or propagate into the sample.
2. Apply continuity of the tangential fields at the surface to get the 2×2 reflection matrix.
3. Normalize it by $r_{ss}$.

Then, at each frequency, fit the complex unknowns $\varepsilon_o$ and $\varepsilon_e$ to the three measured complex ratios, with $\psi$ either known (from the crystal cut or a cross-polarized null) or fitted as a single global parameter across all frequencies. Start from the isotropic inversion of $r_{pp}/r_{ss}$. Use $\theta = \theta_{\text{eff}}$ and the sign convention of §6.1, and check the forward model against the aligned-azimuth closed forms of §7.2 at $\psi = 0°$ and 90°.

The off-diagonal terms are largest near $\psi \approx 45°$ and scale with the birefringence $\varepsilon_e - \varepsilon_o$. Measure away from the aligned azimuths and budget enough averaging time for the off-diagonal elements, which are often small.

### 7.7 Cross-validation and diagnostics

**Isotropic samples in GE mode.** Silicon or gold measured in GE mode should give off-diagonal ratios at the noise floor and an $r_{pp}/r_{ss}$ that agrees with isotropic mode. This tests the whole GE calibration.

**Detecting unexpected in-plane anisotropy.** Off-diagonal elements above the noise in a sample assumed to be isotropic show that the assumption was wrong. Out-of-plane anisotropy cannot be detected this way (§7.1).

**Comparing the two routes.** For in-plane uniaxial samples, compare $\varepsilon_o$ and $\varepsilon_e$ from single-azimuth GE with those from the two-aligned-azimuth isotropic-mode route of §7.2. Agreement validates both the GE calibration and the forward model.

### 7.8 Hardware for probe-polarization switching

**Only optics before the GaP affect the cancellation of $G$.** A half-wave plate (HWP) before the GaP rotates the probe polarization: rotating the plate by 22.5° rotates the polarization by 45°. Any beam deviation it introduces moves the probe relative to the THz focus and changes the overlap, which then differs between probe settings and no longer cancels. Optics after the GaP (quarter-wave plate, Wollaston, photodiodes) only affect balancing, not $G$.

**Balanced detection survives the 45° switch without a second waveplate.** With the quarter-wave plate at 45° to the $\phi = 0°$ probe, rotating the probe by 45° puts it along the quarter-wave plate axis. The EO-induced ellipticity is then converted into a polarization rotation, which the Wollaston, now at 45° to the probe, still reads with full sensitivity and zero static imbalance. The sign of the signal flips, which the calibration absorbs. The isotropic-mode setting (31.7°) needs the quarter-wave plate re-set to 45° to the probe, which is harmless because it sits after the GaP. An alternative is a co-rotating HWP after the GaP.

**Testing the HWP for beam deviation.** A wedge angle $\alpha_w$ deviates the beam by $\delta \approx (n-1)\alpha_w$, and the direction of the deviation rotates with the plate. If the overlap is aligned at one setting, switching by a 22.5° plate rotation displaces the probe at the GaP by $2\,\delta L_{\text{eff}}\sin(11.25°) \approx 0.39\,\delta L_{\text{eff}}$, where $L_{\text{eff}}$ is the distance from the plate to the GaP, or the focal length of any lens that focuses the probe onto the GaP. Two complementary tests:

1. *Direct optical test.* Place a camera or position-sensitive detector after the HWP with a long lever arm and rotate the plate through 360°. A wedged plate makes the spot trace a circle of radius $\delta L$.
2. *THz test on gold.* Record spectra at several HWP angles and divide each by the spectrum at a reference angle. Pure polarization effects change only a real, frequency-independent factor, which may include a sign. Probe walk-off shows up as a high-frequency amplitude roll-off, because the THz focus at the GaP shrinks as $w \propto 1/\omega$, so a fixed offset is a larger fraction of the spot at high frequency. If the path length also changes, it can add a linear phase ramp. Polarization effects repeat every 90° of HWP rotation, whereas wedge deviation repeats only every 360°, which separates the two. Avoid HWP angles where the sensitivity to the gold's reflected polarization passes through zero, because the normalized ratio is noisy there.

A liquid-crystal variable retarder with its axis at 22.5° to the probe can switch between 0 and λ/2 retardance electrically, rotating the polarization by 45° with nothing moving. Its static wedge is then constant and drops out, and its retardance errors are absorbed by the §7.5 calibration.

**Noise budget.** The $E_p$ channel has twice the sensitivity of the $E_s$ channel. Weight the averaging time towards the $s$ channel, or accept the difference in signal-to-noise.

### 7.9 Out-of-plane anisotropy

Optic axes along the surface normal (cleaved layered crystals such as MoS₂, hBN and graphite) need information from outside a single-angle reflection measurement. There are two options:

- **Transmission** at normal incidence through a flake or film measures $\varepsilon_\parallel$; the reflection ρ then gives $\varepsilon_\perp$ through the closed form in §7.2.
- **Variable angle of incidence.** ρ changes with angle differently depending on $\varepsilon_\perp$. Two or more angles determine both components, with $\theta_{\text{eff}}$ calibrated on silicon at each angle.

---

## 8. Sensitivities and error budget

### 8.1 Angle error

If the true effective angle is θ but the inversion uses θ′, the recovered permittivity is an exact affine transformation of the true one:

$$\varepsilon' = A\,\varepsilon + B, \qquad A = \frac{\sin^2\theta'\tan^2\theta'}{\sin^2\theta\tan^2\theta}, \qquad B = \sin^2\theta' - A\sin^2\theta$$

For example, analysing 42° data as 45° gives $A = 1.38$, $B = -0.12$, and silicon comes out at 15.97 instead of 11.68. Near 45° the error is about 5% in $n$ per degree.

Because $A$ and $B$ are real constants, an angle error scales and offsets the spectrum without distorting its shape. Resonance frequencies, damping rates and Drude scattering times are unaffected; oscillator strengths, plasma frequency and $\varepsilon_\infty$ are not. The angle therefore cannot be fitted afterwards from sample data alone, because $A\varepsilon + B$ is just as physically valid as ε. It has to be determined from standards (§5.2).

### 8.2 Conditioning for strong conductors

$$\frac{\partial\varepsilon}{\partial\rho} = -\frac{4\sin^2\theta\tan^2\theta\,(1-\rho)}{(1+\rho)^3}$$

As $\rho \to -1$ (highly conducting samples), errors in ρ are strongly amplified. Strongly metallic samples will be noisier at 45° than dielectrics or semiconductors.

### 8.3 Other contributions

| Source | Effect | Mitigation |
|---|---|---|
| Magnet polarization offset $\delta$ | Mixing of $p$ and $s$, error of a few % per degree | Wire-grid null (§4.4); fitted as $\beta_0$ in GE calibration |
| Out-of-plane sample tilt | Rotates the $p/s$ basis | THz-peak tilt scan on both axes (§4.5) |
| Drift between states | Does not cancel in ratios | Interleaved cycling; power-monitor normalization |
| Probe-angle and crystal-axis offsets (GE) | Off-diagonal leakage in $\mathbf{D}$ | Magnet-sweep calibration (§7.5) |
| HWP beam deviation (GE) | $G$ differs between probe settings | Deviation tests (§7.8); low-wedge plate or liquid-crystal retarder |
| Waveplate retardance error (GE) | Wrong projection weights | Included in the magnet-sweep fit |
| Static GaP birefringence and strain | Offset in the detection; extinction floor for GE | ±M differencing; good-quality, low-strain crystal |
| Finite focusing cone | ρ averaged over a range of angles; material-dependent bias | Fitting $\theta_{\text{eff}}$ absorbs most of it; moderate focusing (f/2 or slower) |
| Roughness with $\sigma$ approaching $\lambda/10$ | Polarization-dependent diffuse loss | Watch the high-frequency end |
| Stray field of the emitter magnet at the sample | Magneto-optic off-diagonal terms that change between magnet states | Check with a gaussmeter at the sample position |

---

## 9. Workflow summary

**One-time and periodic setup**

1. Mount the GaP with $[\bar 1 10]$ along $p$ and $[001]$ along $s$.
2. Set the magnet zero by the wire-grid null, then remove the grid.
3. Test the HWP for beam deviation (§7.8).
4. GE calibration: magnet sweep on gold at both probe settings, giving $\mathbf{D}$ and $E_p/E_s$.
5. Isotropic-mode calibration: gold at $\phi \approx 31.7°$, giving κ.
6. Silicon: obtain $\theta_{\text{eff}}$, check $\varepsilon_{\text{Si}}$, and run the tilt test.
7. Silicon in GE mode as a check: off-diagonal terms at the noise floor.

**Per sample**

1. Mount the sample, centre its tilt on the THz peak, and leave the downstream optics untouched.
2. *Known isotropic sample:* acquire in isotropic mode (magnet 0/90/180/270° at $\phi \approx 31.7°$), compute $\rho = R/\kappa$, and invert (§6).
3. *Possibly anisotropic, or optic axis in-plane:* acquire in GE mode (magnet sequence at both probe settings). Extract the normalized $J$ (§7.4) and fit $\varepsilon_o$, $\varepsilon_e$ (§7.6). Optionally cross-check with isotropic mode at two aligned azimuths (§7.2).
4. *Optic axis along the normal:* combine reflection ρ with $\varepsilon_\parallel$ from transmission (§7.2, §7.9).

---

## 10. Key references

- T. Nagashima and M. Hangyo, "Measurement of complex optical constants of a highly doped Si wafer using terahertz ellipsometry," *Applied Physics Letters* (2001).
- M. Neshat and N. P. Armitage, "Developments in THz range ellipsometry," *Journal of Infrared, Millimeter, and Terahertz Waves* (2013).
- P. C. M. Planken et al., "Measurement and calculation of the orientation dependence of terahertz pulse detection in ZnTe," *JOSA B* (2001).
- N. C. J. van der Valk, W. A. M. van der Marel and P. C. M. Planken, "Terahertz polarization imaging," *Optics Letters* (2005).
- T. Seifert et al., "Efficient metallic spintronic emitters of ultrabroadband terahertz radiation," *Nature Photonics* (2016).
- D. W. Berreman, "Optics in stratified and anisotropic media: 4×4-matrix formulation," *JOSA* (1972).
- M. Schubert, "Polarization-dependent optical parameters of arbitrarily anisotropic homogeneous layered systems," *Physical Review B* (1996).
- R. M. A. Azzam and N. M. Bashara, *Ellipsometry and Polarized Light* (North-Holland), for sign conventions and the general inversion.
