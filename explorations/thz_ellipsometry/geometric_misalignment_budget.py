"""Three geometric misalignments that look alike and behave differently.

Samuel's question: is an out-of-plane sample tilt the same thing as an emitter-angle offset?
Almost -- both rotate the p/s frame and both produce a Moebius transform of rho rather than a
scale factor, so neither is removed by a scalar calibration. But they differ in origin, in how
they scale with incidence angle, and in how they are fixed:

  IN-PLANE angle error    theta is simply wrong. NO polarisation mixing. Fittable from HR-Si.
  EMITTER offset          a FIXED instrument property: the magnet's zero versus the plane of
                          incidence. Same for every sample -> measure once, correct forever.
  OUT-OF-PLANE tilt       rotates the SAMPLE's Jones matrix. Varies per mount -> must be made
                          common between sample and reference, or measured per sample.

The headline: the emitter offset is a nuisance, differential out-of-plane tilt is the dangerous
one, and random surface unevenness is nearly harmless because the off-diagonal terms are odd in
the tilt and cancel for a symmetric distribution.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# This exploration sits two levels below the repo root; the production package lives there.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ellipsometry import materials, model  # noqa: E402

__all__ = ["tilted_jones", "averaged_jones", "recovered_index", "main"]


def tilted_jones(index_sample, incidence_angle_rad, out_of_plane_tilt_rad):
    """Lab-frame Jones matrix of an isotropic sample tilted out of the plane of incidence."""
    reflection_p, reflection_s = model.reflection_coefficients(index_sample, incidence_angle_rad)
    cosine, sine = np.cos(out_of_plane_tilt_rad), np.sin(out_of_plane_tilt_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    diagonal = np.array([[reflection_p, 0.0], [0.0, reflection_s]], dtype=complex)
    return rotation @ diagonal @ rotation.T


def averaged_jones(index_sample, incidence_angle_rad, tilt_rms_rad, points=41):
    """Jones matrix averaged over a SYMMETRIC distribution of local facet tilts.

    This is an uneven surface rather than a tilted one. The off-diagonal terms are odd in the
    tilt, so they cancel in the mean and only a second-order diagonal correction survives.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(points)
    weights = weights / weights.sum()
    return sum(weight * tilted_jones(index_sample, incidence_angle_rad, tilt_rms_rad * node)
               for node, weight in zip(nodes, weights))


def recovered_index(sample_jones, reference_jones, incidence_angle_rad, index_sample,
                    channel_ratio=1.0, emitter_offset_rad=0.0, frequency_hz=1e12):
    """Measure both, calibrate the scalar channel ratio on the reference, invert the sample."""
    def measured(jones):
        upper = channel_ratio * jones[0, 0] + jones[1, 0]
        lower = channel_ratio * jones[0, 1] + jones[1, 1]
        cosine, sine = np.cos(emitter_offset_rad), np.sin(emitter_offset_rad)
        return (upper * cosine + lower * sine) / (-upper * sine + lower * cosine)

    gold = materials.gold_index(frequency_hz)
    estimated_ratio = measured(reference_jones) / model.ellipsometric_ratio(
        gold, incidence_angle_rad)
    return model.index_from_ellipsometric_ratio(
        measured(sample_jones) / estimated_ratio, incidence_angle_rad,
        reference_index=index_sample)


def _error(sample_tilt_deg, reference_tilt_deg, angle_deg, emitter_offset_deg=0.0,
           random=False):
    angle = np.deg2rad(angle_deg)
    silicon = materials.high_resistivity_silicon_index(1e12)
    gold = materials.gold_index(1e12)
    builder = averaged_jones if random else tilted_jones
    sample = builder(silicon, angle, np.deg2rad(sample_tilt_deg))
    reference = builder(gold, angle, np.deg2rad(reference_tilt_deg))
    recovered = recovered_index(sample, reference, angle, silicon,
                                emitter_offset_rad=np.deg2rad(emitter_offset_deg))
    return abs(recovered - silicon)


def main(angles_deg=(45.0, 55.0, 70.0, 75.0), tolerance=0.02):
    print(f"Index error |dN| from geometric misalignment, HR-Si at 1 THz "
          f"(tolerance {tolerance})\n")

    blocks = [
        ("A. OUT-OF-PLANE tilt on the SAMPLE only (reference flat)",
         (0.1, 0.25, 0.5, 1.0, 2.0),
         lambda value, angle: _error(value, 0.0, angle)),
        ("B. the SAME tilt on sample and reference (a shared mount error)",
         (0.25, 0.5, 1.0, 2.0),
         lambda value, angle: _error(value, value, angle)),
        ("C. EMITTER angular offset only (both flat)",
         (0.1, 0.25, 0.5, 1.0, 2.0),
         lambda value, angle: _error(0.0, 0.0, angle, emitter_offset_deg=value)),
        ("D. RANDOM surface unevenness, rms (both surfaces)",
         (0.25, 0.5, 1.0, 2.0),
         lambda value, angle: _error(value, value, angle, random=True)),
    ]
    for title, values, evaluate in blocks:
        print(title)
        print(f"   {'deg':>8}" + "".join(f"{f'theta={a:g}':>12}" for a in angles_deg))
        for value in values:
            row = f"   {value:>8.2f}"
            for angle_deg in angles_deg:
                row += f"{evaluate(value, angle_deg):>12.4f}"
            print(row)
        print()

    print("E. TOLERANCE: largest misalignment that keeps |dN| inside the tolerance")
    print(f"   {'term':>34}" + "".join(f"{f'theta={a:g}':>12}" for a in angles_deg))
    terms = [("differential out-of-plane tilt [deg]",
              lambda value, angle: _error(value, 0.0, angle)),
             ("shared out-of-plane tilt [deg]",
              lambda value, angle: _error(value, value, angle)),
             ("emitter angular offset [deg]",
              lambda value, angle: _error(0.0, 0.0, angle, emitter_offset_deg=value))]
    for name, evaluate in terms:
        row = f"   {name:>34}"
        for angle_deg in angles_deg:
            grid = np.arange(0.005, 3.0, 0.005)
            errors = np.array([evaluate(value, angle_deg) for value in grid])
            inside = grid[errors <= tolerance]
            row += f"{(inside.max() if inside.size else 0.0):>12.3f}"
        print(row)
    print("\n   -> differential tilt is the dangerous term and it strongly favours a HIGH")
    print("      incidence angle; the emitter offset is mild and favours a low one.")


if __name__ == "__main__":
    main()
