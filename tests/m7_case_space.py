"""Seeded random M7 cells spanning the designed-case space of the parity suite.

``random_cases`` draws mixed HAM populations of the kind the designed cases in
``test_m7_step_all.py`` sample by hand -- every soluble and insoluble mode
independently populated, empty or near the ``CMIN_AERNL`` gate -- with
per-particle masses built from a count-median dry radius inside each mode's
size range, so the populations are geometrically consistent. Ranges are wider
than the designed cases on purpose (cold upper troposphere to warm surface,
fresh nucleation bursts to aged coarse dust) so the float32 property tests see
the extreme number contrasts where cancellation matters.

Plain numpy, float64; callers cast to the working dtype. Not a test module
(no ``test_`` prefix), so pytest only imports it.
"""
import numpy as np

from m7_jax.constants import (AVOGADRO, CMR2RAM, PI, SO4_MOLAR_MASS, SPECIES_DENSITY,
    COMPONENT_MODE, COMPONENT_SPECIES)

# Count-median dry radius range [cm] and log-uniform number range [cm-3] per
# mode NS KS AS CS KI AI CI.
_RADIUS = ((5e-8, 5e-7), (5e-7, 5e-6), (5e-6, 5e-5), (5e-5, 3e-4),
           (5e-7, 5e-6), (5e-6, 5e-5), (5e-5, 3e-4))
_NUMBER = ((1e-1, 1e6), (1e-1, 1e4), (1e-2, 2e3), (1e-4, 10.),
           (1e-4, 1e4), (1e-4, 10.), (1e-5, 2.))


def _loguniform(rng, low, high, size):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def random_cases(n, seed):
    """Return a dict of ``n`` random cells (same keys as the designed cases)."""
    rng = np.random.default_rng(seed)
    number = np.zeros((n, 7))
    for mode, (low, high) in enumerate(_NUMBER):
        value = _loguniform(rng, low, high, n)
        draw = rng.uniform(size=n)
        value = np.where(draw < 0.25, 0., value)                                # empty mode
        value = np.where((draw >= 0.25) & (draw < 0.32), _loguniform(rng, 1e-11, 1e-6, n), value)
        number[:, mode] = value
    mass = np.zeros((n, 18))
    for mode in range(7):
        radius = _loguniform(rng, *_RADIUS[mode], n)
        volume = (4 * PI / 3) * (radius * CMR2RAM[mode]) ** 3            # cm3 per particle
        members = [jn for jn in range(18) if COMPONENT_MODE[jn] == mode]
        share = rng.dirichlet(np.ones(len(members)), n)
        for k, jn in enumerate(members):
            species = COMPONENT_SPECIES[jn]
            grams = share[:, k] * volume * SPECIES_DENSITY[species] * 1e-3  # g per particle
            if species == 0:
                mass[:, jn] = grams / (SO4_MOLAR_MASS / AVOGADRO) * number[:, mode]
            else:
                mass[:, jn] = grams * number[:, mode] * 1e12                  # ug m-3
    production = np.where(rng.uniform(size=n) < 0.2, 0., _loguniform(rng, 1e-1, 1e6, n))
    cloud = np.where(rng.uniform(size=n) < 0.5, 0., rng.uniform(size=n))
    return dict(
        mass=mass, number=number,
        temperature=rng.uniform(190., 310., n),
        pressure=rng.uniform(1e4, 1.05e5, n),
        relative_humidity=rng.uniform(0., 1., n),
        gas=_loguniform(rng, 1e3, 1e9, n),
        production=production,
        cloud_fraction=cloud,
        timestep=rng.uniform(60., 2400., n),
        forest_fraction=rng.uniform(0., 1., n),
        in_pbl=rng.uniform(size=n) < 0.5,
    )
