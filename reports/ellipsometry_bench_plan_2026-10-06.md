# THz ellipsometry — bench plan, 6–7 Oct 2026 (two days)

Goal of the two days: validate isotropic-mode ellipsometry on silicon (Phase 2 of
`docs/THZ_ELLIPSOMETRY_IMPLEMENTATION_PLAN.md`), and collect the half-wave-plate (HWP)
beam-deviation dataset. Day 1 holds the long runs; day 2 holds the shorter ones.

Tools referenced below come from `thz_ellipsometry_bench_run_me.py` (being built 2026-10-05).
Run them on the data directory as files arrive.

**Assumed timings, to be replaced by measured ones:** ~2 min per scan (from an earlier `.acc`
file); initial purge ~3.7 h to settle (F32); purge after a short box opening: unknown. Step 1.6
measures it.

---

## Filename convention (manual for now)

```
<sample>_mag=<angle READ off the scale, deg>_cyc=<nn>.acc     doped-si_mag=090_cyc=01.acc
ref-gold_mag=270_cyc=01.acc            "gold" or "mirror" in the name = channel reference
angleref-hrsi_mag=000_cyc=01.acc       only when the angle is to be FITTED on HR-Si
..._probe=15.acc                       only for a probe setting other than the day's main one
wgp_grid=s_null=0_mag=086.acc          wire-grid null sweep: grid orientation, the polarisation
                                       this null defines (0/90/180/270), magnet reading
hwp-gold_hwp=090_mag=045.acc           HWP test series (hwp = plate angle READ, deg)
knife_axis=x_pos=12.30.acc             knife-edge (record the units in the log)
```

- **Write the angle you actually read, not the one you aimed for.** The fit accepts any angles.
  Use `m` for minus (`mag=m30`).
- One file = one setting = several repeat scans. **At least 4 scans** per file; the noise model
  needs 3.
- Acquisition order comes from the header timestamps, so `cyc` only keeps names unique.
- Keep one directory per block (see below).

## Nesting order (minimises box openings, preserves alignment)

```
sample/reference block   (outermost: a remount is the least reproducible change)
  probe setting (HWP)    (all magnet states at ONE probe setting: walk-off cancels within it)
    magnet state         (innermost: changes most often)
```

Why drift between blocks does not matter: each block is self-referenced. Gold gives C as a ratio
inside the gold block, the sample gives rho inside its block, so the purge state between blocks
cancels. **Only drift inside a block matters**, so for every magnet change inside a block:

- **Make every disturbance identical:** same open time, same wait before the first scan, same
  number of scans. The purge transient then becomes the same function of time-since-closing for
  every state, which is common-mode, not differential.
- **Order the states as a palindrome when time allows:** 0, 90, 180, 270, 270, 180, 90, 0. A
  linear drift then cancels by symmetry, and the drift fit gets leverage. A single pass of
  0/90/180/270 is the minimum: four states are needed for the background term plus one drift
  parameter.

---

## Day 1 (Tue 6 Oct) — alignment, then the long silicon runs

### 1. Box open: everything that needs hands inside, before the long purge

- [ ] **1.1 Laser warm-up + knife-edge test** (unpurged is fine; water lines only remove a few
      frequencies). Scan the THz beam at the sample plane on both axes. Do it with and without
      the 5 mm aperture if time allows (OQ12: does removing it shrink the low-frequency spot?).
      Record: blade axis, positions and units, distance from the focusing optic, aperture yes/no.
      *(Analysis is deferred: decide afterwards whether a blur correction is worth building.)*
- [ ] **1.2 GaP to ~31 deg from [001].** Not exactly balanced is fine: gold calibrates any C.
      **Stay more than ~3 deg from 0 and 45 deg** (degenerate: one channel blind). Note the angle
      set; it goes in `config["detection"]["probe_azimuth_deg"]`. Do not touch the crystal or the
      probe again until day 2's HWP work.
- [ ] **1.3 Quick balance check on gold:** two short scans, magnet at p and at s. Both pulses must
      be clearly above noise. The amplitude ratio is roughly |C|; anything within ~3x is workable.
- [ ] **1.4 Magnet calibration with the wire grid (WGP): all FOUR states, not just p.** Per-state
      errors of the magnet scale cannot be fitted from the sample data (checked: no signature in
      the fit), and a polarisation error MIXES the channels rather than scaling them, so gold does
      not absorb it. The grid is the only place they are measured.
      - Grid passing **s**: sweep through the null near the p reading (`null=0`) and the one
        180 deg away (`null=180`).
      - Grid passing **p**: sweep through `null=90` and `null=270`.
      - Each sweep: readings at about -20, -8, -4, -2, +2, +4, +8, +20 deg around the null, 3+
        scans each (~30 s per point is plenty; this is about the sign change, not the spectrum).
      - Run `bench_run_me null <dir>`. Coherent detection makes the signal change SIGN through the
        null, so it is found to a fraction of a degree. Sweeps 180 deg apart are fitted together:
        the background does not reverse with the magnet and the signal does, which separates a
        background from a moved null (a lone sweep cannot: 2% background looks like a 1.1 deg
        shift). It prints the line to paste: `config["geometry"]["magnet_calibration"] = {...}`.
      - The tool also prints the spacing between nulls minus 90 deg. A large value means a scale
        or sense error of the magnet mount; note it.
- [ ] **1.5 Remove the grid.** From here on, set the magnet only at the four calibrated readings,
      read the scale the same way each time, and record what you read.
- [ ] **1.6 Stray field:** gaussmeter at the sample position, magnet at 0 and 90 (plan sec. 8.3).
      Record the values.
- [ ] **1.7 Mount gold.** Set the incidence angle mechanically (visible laser, 45 deg); note how,
      and your uncertainty estimate -> `geometry.incidence_angle_uncertainty_deg`. Centre the tilt
      on the THz peak, both axes (plan sec. 4.5). Do not re-steer anything after the sample.
- [ ] **1.8 Optional: HWP optical wedge test** while the box is open. Camera or PSD after the HWP
      with a long lever arm; rotate the plate through 360 deg. A wedged plate traces a circle; note
      its radius and the lever arm.

### 2. Close the box: long initial purge

- [ ] **2.1 Start `bench_run_me live <gold dir>` with repeated scans at one magnet state.** It
      reports the pulse delay and amplitude against time, and the drift rate in fs/min.
      Equilibrated = drift rate below ~1 fs/min over the last 3 scans (to be tuned on the day).
- [ ] **2.2 Measure the short-disturbance settling time T_eq:** once settled, open the box for the
      time a magnet rotation takes, close it, and watch the live check settle again. **T_eq, and
      the open time, become fixed for every magnet change today.**

### 3. Main silicon dataset (one directory per block; magnet order per the nesting above)

At ~2 min/scan with 4–6 scans per state plus T_eq per state, a single-pass block is roughly
4 x (T_eq + 10 min). Do the blocks in this order: gold brackets the day.

- [ ] **Block A — gold:** 0, 90, 180, 270 (palindrome if time). `ref-gold_...`
- [ ] **Block B — doped Si:** palindrome. **The key sample**: it is the real test, and the
      dispersive one that lets the tilt be fitted. Polished: align the angle with visible light.
- [ ] **Block C — HR-Si:** 0, 90, 180, 270. A consistency check; it needs good mounting, because a
      flat sample's tilt cannot be fitted out.
- [ ] **Block D — gold again**, same as A. Tells us whether C drifts over the day.
- [ ] **After each block:** `thz_ellipsometry_run_me.py --directory <block dir>` (gold blocks
      need a sample; run B and C each against A, then against D). Check the `[check_assumptions]`
      findings before moving on. A large `harmonic_model_holds` chi-square means something
      changed inside the block.

**Log for every file:** the magnet reading actually set, the time the box was closed, anything
touched. A paper log is fine.

The analysis now also fits a per-acquisition AMPLITUDE ramp (laser power, broadband purge loss),
alongside the delay ramp; both are reported, and a change of more than 5% across a block is flagged.

---

## Day 2 (Wed 7 Oct) — HWP dataset, then shorter runs

### 4. HWP beam-deviation test on gold (plan sec. 7.8): the question you have wanted answered

Magnet fixed at one state that gives both channels signal (e.g. 45 deg pol), gold mounted, every
setting with the same open/T_eq/scans routine. `hwp-gold_hwp=<read>_mag=045.acc`.

- [ ] **Plate at 0, 90, 180, 270 deg:** the SAME polarisation state four times. Polarisation
      repeats every 90 deg of plate rotation; wedge walk repeats only every 360 deg. So ANY spread
      between these four is beam deviation: a high-frequency amplitude roll-off, and maybe a
      linear phase ramp.
- [ ] **Plate at 22.5 and 45 deg:** what the polarisation switch itself does (the probe moves
      45 deg and 90 deg).
- [ ] `bench_run_me hwp <dir>` reports roll-off slope and delay versus plate angle, and separates
      the 360-deg (wedge) part from the 90-deg (polarisation) part.
- [ ] **Return the plate to the day-1 angle** (and re-set the QWP to 45 deg to the probe if it was
      moved: QWP and Wollaston sit after the GaP, so they affect balancing, not the THz path).

### 5. Shorter runs (each one block of 0/90/180/270)

- [ ] **5.1 Probe-rotation calibration:** doped Si at a second probe setting about 15 deg away
      from the main one (not near 0 or 45 from [001]), with `_probe=<deg>` in the name. Tests the
      no-gold calibration against block A's gold.
- [ ] **5.2 Remount repeatability (gate V8):** remove and remount doped Si, then repeat block B
      single-pass; twice if time allows. Target: < 1% change.
- [ ] **5.3 Angle-pinning test (plan sec. 5.3):** HR-Si tilted +1 and -1 deg in-plane, one block
      each. Run with `calibration.incidence_angle = "fit_from_reference"`. If the fitted angle does
      not move, the cones are matched.
- [ ] **5.4 Optional: c-cut sapphire**, one block. Back-face echo must be windowed out. Pair it
      later with normal-incidence transmission for the out-of-plane route (in-plane eps ~9.4,
      out-of-plane ~11.6).
- [ ] **5.5 Transmission of the doped wafer** (truth standard), if it can be done on this bench
      or slotted in elsewhere.

---

## Things to bring back for the analysis

- Every `.acc` file, one directory per block.
- The log: magnet readings, box closures, T_eq, GaP angle, incidence-angle method and
  uncertainty, gaussmeter values, knife-edge geometry, wedge-circle radius.
- Which doped wafer it is, and its resistivity if known.
