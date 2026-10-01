"""Reference optical constants for the ellipsometry validation samples.

Kept in one place so the validation targets are not scattered through the code, and so every
number carries its source. All indices follow the package convention ``N = n - i*k``, k >= 0.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SILICON_HIGH_RESISTIVITY_INDEX",
    "SILICON_STATIC_PERMITTIVITY",
    "doped_silicon_index",
    "extinction_from_power_absorption",
    "gold_index",
    "high_resistivity_silicon_index",
    "power_absorption_from_extinction",
]

ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
VACUUM_PERMITTIVITY = 8.8541878128e-12
SPEED_OF_LIGHT = 2.99792458e8

#: Float-zone high-resistivity silicon, THz refractive index. Flat to <1e-3 across 0.1-4 THz;
#: the standard reference value used throughout the THz literature.
SILICON_HIGH_RESISTIVITY_INDEX = 3.4175

#: Static (lattice) permittivity of silicon, = n^2 for the undoped crystal.
SILICON_STATIC_PERMITTIVITY = SILICON_HIGH_RESISTIVITY_INDEX**2

#: Conservative bound on the power absorption of HR float-zone silicon at 1 THz, in /cm.
#: Published values for >10 kohm.cm material sit at or below this.
SILICON_HIGH_RESISTIVITY_ABSORPTION_PER_CM = 0.05

#: Drude parameters for electrons in silicon.
SILICON_ELECTRON_EFFECTIVE_MASS_RATIO = 0.26
SILICON_ELECTRON_MOBILITY_CM2_PER_VS = 1400.0


def extinction_from_power_absorption(absorption_per_cm, frequency_hz):
    """k from the power absorption coefficient, alpha = 4*pi*k/lambda."""
    wavelength_cm = SPEED_OF_LIGHT / _positive_frequency(frequency_hz) * 100.0
    return np.asarray(absorption_per_cm, dtype=float) * wavelength_cm / (4.0 * np.pi)


def power_absorption_from_extinction(extinction, frequency_hz):
    wavelength_cm = SPEED_OF_LIGHT / np.asarray(frequency_hz, dtype=float) * 100.0
    return 4.0 * np.pi * np.asarray(extinction, dtype=float) / wavelength_cm


def high_resistivity_silicon_index(frequency_hz,
                                   absorption_per_cm=SILICON_HIGH_RESISTIVITY_ABSORPTION_PER_CM):
    """HR silicon index. k is ~1e-4 at 1 THz -- genuinely below any reflection measurement.

    Recovering ``k = 0`` on this sample is the CORRECT answer, not a failure of the method.
    """
    return (SILICON_HIGH_RESISTIVITY_INDEX
            - 1j * extinction_from_power_absorption(absorption_per_cm, frequency_hz))


def _positive_frequency(frequency_hz):
    """Replace the DC bin with the smallest positive frequency present.

    An rfft axis always starts at zero, where every Drude model diverges. Clamping keeps the
    array shape and stops the divergence from spraying invalid-value warnings through the run;
    the DC bin is masked out of the analysis anyway.
    """
    frequencies = np.asarray(frequency_hz, dtype=float)
    positive = frequencies[frequencies > 0]
    floor = positive.min() if positive.size else 1.0
    return np.where(frequencies > 0, frequencies, floor)


def doped_silicon_index(frequency_hz, resistivity_ohm_cm,
                        mobility_cm2_per_vs=SILICON_ELECTRON_MOBILITY_CM2_PER_VS,
                        effective_mass_ratio=SILICON_ELECTRON_EFFECTIVE_MASS_RATIO):
    """Drude index of doped silicon from its DC resistivity.

    eps(w) = eps_lattice - w_p^2 / (w^2 - i*w/tau), with the carrier density implied by the
    resistivity and mobility.  Written in the N = n - i*k convention, so Im(eps) < 0.
    """
    angular_frequency = 2.0 * np.pi * _positive_frequency(frequency_hz)
    conductivity_si = 100.0 / float(resistivity_ohm_cm)          # S/m from ohm.cm
    mobility_si = mobility_cm2_per_vs * 1e-4
    carrier_density = conductivity_si / (ELEMENTARY_CHARGE * mobility_si)
    effective_mass = effective_mass_ratio * ELECTRON_MASS
    scattering_time = mobility_si * effective_mass / ELEMENTARY_CHARGE
    plasma_squared = (carrier_density * ELEMENTARY_CHARGE**2
                      / (VACUUM_PERMITTIVITY * effective_mass))
    permittivity = SILICON_STATIC_PERMITTIVITY - plasma_squared / (
        angular_frequency**2 - 1j * angular_frequency / scattering_time)
    index = np.sqrt(np.asarray(permittivity, dtype=complex))
    return np.where(index.imag > 0.0, -index, index)


def doped_silicon_carrier_density(resistivity_ohm_cm,
                                  mobility_cm2_per_vs=SILICON_ELECTRON_MOBILITY_CM2_PER_VS):
    """Carrier density in m^-3 implied by a DC resistivity."""
    return (100.0 / float(resistivity_ohm_cm)) / (
        ELEMENTARY_CHARGE * mobility_cm2_per_vs * 1e-4)


#: DC conductivity of gold, S/m. The THz value is within a few percent of this.
GOLD_DC_CONDUCTIVITY_SI = 4.1e7
GOLD_SCATTERING_TIME_S = 27e-15


def gold_index(frequency_hz, dc_conductivity_si=GOLD_DC_CONDUCTIVITY_SI,
               scattering_time_s=GOLD_SCATTERING_TIME_S):
    """Drude index of gold. |N| ~ 900 at 1 THz, so rho_gold = -1 to about 0.1%.

    That is why gold is the right channel-ratio reference and the wrong angle reference:
    rho is essentially -1 no matter what the incidence angle is.
    """
    angular_frequency = 2.0 * np.pi * _positive_frequency(frequency_hz)
    plasma_squared = dc_conductivity_si / (VACUUM_PERMITTIVITY * scattering_time_s)
    permittivity = 1.0 - plasma_squared / (
        angular_frequency**2 - 1j * angular_frequency / scattering_time_s)
    index = np.sqrt(np.asarray(permittivity, dtype=complex))
    return np.where(index.imag > 0.0, -index, index)


#: Named reference materials the driver can request by string, so a config dict never has to
#: carry a complex number. Each entry is a callable of frequency.
REFERENCE_MATERIALS = {
    "gold": gold_index,
    "hr_silicon": high_resistivity_silicon_index,
}


def reference_index(name, frequency_hz):
    """Look up a named reference material. Raises with the valid names if unknown."""
    try:
        builder = REFERENCE_MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"unknown reference material {name!r}; "
            f"known names are {sorted(REFERENCE_MATERIALS)}"
        ) from None
    return builder(frequency_hz)
