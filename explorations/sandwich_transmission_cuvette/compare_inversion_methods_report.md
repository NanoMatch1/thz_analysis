# Sandwich-transmission inversion comparison (shared thz_core front-end)

Dataset: Vasilis_Data 200 K triplet (air / empty cuvette / filled cuvette).  One reliable front-end (thz_core in-place windowing) feeds the SAME measured transfer functions to every inversion method, isolating the inversion math.

## Front-end (shared)

- timestep: 0.0500 ps;  band: 0.8-2.0 THz, 246 bins
- raw peak delay substrate vs air: 5.30 ps (2-window path 1.8 mm → n_sub ≈ 1.88)
- raw peak delay sample vs substrate: 0.50 ps (sample 0.13 mm → n_sample ≈ 2.15)

## 1. Substrate refractive index  n_sub(omega)

Fabry-Perot OFF (gated) for all grid methods: the mm-scale windows are gatable.

| method | n_sub median | n_sub range | k_sub median |
|---|---|---|---|
| thz_core analytic (free-standing)  | 1.949 | [1.945, 1.963] | 0.0000 |
| thz_core grid (ASASA, FP off)      | 1.950 | [1.937, 1.972] | 0.0000 |
| MATLAB-port grid (FP off)          | 1.950 | [1.937, 1.969] | 0.0000 |
| Novelli analytic (ns=1)            | 1.949 | [1.945, 1.963] | -0.0092 |

## 2. Sample complex refractive index  n_hat = n + i*k

| method | n median | n range | k median |
|---|---|---|---|
| thz_core analytic (free-standing)  | 2.151 | [2.112, 2.219] | 0.1249 |
| thz_core grid (ASMSA, FP off)      | 2.146 | [2.106, 2.218] | 0.1910 |
| thz_core grid (ASMSA, FP on)       | 2.156 | [2.088, 2.202] | 0.1860 |
| MATLAB-port grid (FP off)          | 2.145 | [2.106, 2.217] | 0.1912 |
| MATLAB-port grid (FP on)           | 2.157 | [2.088, 2.202] | 0.1868 |
| Novelli analytic                   | 2.146 | [2.105, 2.217] | 0.1905 |

## 3. Sample complex conductivity  sigma (S/m), eps_background = 1 (vacuum)

| method | sigma_real median | sigma_real range | sigma_imag median |
|---|---|---|---|
| thz_core analytic (free-standing)  | 4.190e+01 | [0.00e+00, 1.01e+02] | -2.757e+02 |
| thz_core grid (ASMSA, FP off)      | 6.357e+01 | [1.51e+01, 1.22e+02] | -2.724e+02 |
| thz_core grid (ASMSA, FP on)       | 6.248e+01 | [1.48e+01, 1.22e+02] | -2.819e+02 |
| MATLAB-port grid (FP off)          | 6.365e+01 | [1.82e+01, 1.22e+02] | -2.725e+02 |
| MATLAB-port grid (FP on)           | 6.272e+01 | [1.82e+01, 1.22e+02] | -2.817e+02 |
| Novelli analytic                   | 6.343e+01 | [1.84e+01, 1.21e+02] | -2.722e+02 |

## 4. Critical comparison & recommendation

**On a shared, reliable front-end the inversion *models* agree where they should and disagree only where the physics demands it.**

- **Substrate n_sub:** all four methods → 1.95 (fused silica). With the clean front-end the grid solvers no longer rail/sawtooth (only a cosmetic ±grid-step ripple remains); the old 2.2-2.5 inflation was purely the MATLAB/legacy padding. Novelli needs the absolute-phase anchor (its sole weak point — without it n tilts and reads 1.71); given the anchor it matches.

- **Sample n:** all six methods cluster at 2.15 ± 0.01. n is INSENSITIVE to the geometry treatment — free-standing, sandwich grid, and Novelli all agree, because the filled/empty differential cancels the substrate phase.

- **Sample k / conductivity — THE decisive split:** the free-standing analytic gives k ≈ 0.125, sigma_real ≈ 42 S/m; every sandwich-aware method (grid ASMSA, Novelli with ns = n_sub) gives k ≈ 0.19, sigma_real ≈ 63 S/m — ~50 % higher. Raising the grid k-ceiling to 0.4 did not move it, so 0.19 is real, not clipping. The free-standing model mis-attributes the silica/sample Fresnel transmission to absorption; the sandwich models account for the n≈1.95 surround and recover the correct k. **For absorption / conductivity you MUST use a sandwich-aware inversion.**

- **Fabry-Perot:** on vs off shifts sigma_real ~2 % and sigma_imag ~3 % here — a small correction for this gatable 0.13 mm sample. Keep it opt-in; the textbook 1/(1 - r^2 P^2) sign is correct (the substrate MATLAB sign-slip is moot — substrate is gated).

- **Grid vs Novelli:** the grid ASMSA and Novelli sandwich now agree to ~1e-3 in n and ~1e-3 in k. Novelli is faster and analytic but hinges entirely on the phase anchor; the grid is more robust (handles FP, complex n_sub, branch continuity) at higher cost.

**Recommendation:** continue with the **thz_core front-end + thz_core grid substrate_sandwich (ASMSA) inversion, FP opt-in**. It gives the correct substrate n, the correct sample n AND k/conductivity, handles the FP correction and a complex n_sub natively, and is the maintained codebase. Keep Novelli (with the anchor) as a fast analytic cross-check; retire the free-standing analytic for sample k/sigma (n-only is fine). Retire the MATLAB/legacy asymmetric-padding front-ends entirely.
