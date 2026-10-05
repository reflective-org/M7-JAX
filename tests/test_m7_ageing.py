"""Insoluble-to-soluble ageing (``m7_concoag``/``m7_coat``) parity.

Coating thresholds (zero/below/at/above one monolayer) and zero/populated
insoluble modes are the specific risk this module carries: ``m7_coat``'s
``so4_coating_threshold`` monolayer test is a hard ``MIN``/gate, not a smooth
function, so any orientation or indexing slip shows up as a wrong regime
rather than a small numeric drift.
"""
import jax
import numpy as np
import pytest

from m7_jax.coagulation import _coat, dnum_all
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update("jax_enable_x64", True)

NATIVE_SO4 = 98.08 / 6.022e23


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    executable, _ = build_oracle(tmp_path_factory.mktemp("m7-ageing"), checked=True)
    return executable


def cases(monolayers):
    """One cell per requested coating fraction (0 = zero zso4, else a

    multiple of the exact ``m7_coat`` critical constant for that cell).
    Coagulation is switched off so every number/mass change is attributable
    to concoag/coat alone (m7_delcoa calls m7_concoag unconditionally, so
    ageing still proceeds from the insoluble_sulfate input with coagulation
    off, exactly as in the native routine).
    """
    n = len(monolayers)
    number = np.zeros((n, 7))
    number[:, 0] = 2e4
    number[:, 1] = 300
    number[:, 2] = 30
    number[:, 3] = 0.2
    number[:, 4] = 50   # KI
    number[:, 5] = 5     # AI
    number[:, 6] = 0.5   # CI
    radius = np.tile([2e-7, 2e-6, 1e-5, 1e-4, 3e-6, 2e-5, 1.5e-4], (n, 1))
    density = np.tile([1.841, 1.7, 1.4, 1.2, 1.8, 2.6, 2.6], (n, 1))

    mass = np.zeros((n, 18))
    for mode, idx in ((0, 0), (1, 1), (2, 2), (3, 3)):
        mass[:, idx] = number[:, mode] * (4 * np.pi / 3) * radius[:, mode] ** 3 * density[:, mode] / NATIVE_SO4
    mass[:, 4] = number[:, 1] * 1e-15 * 1e12
    mass[:, 5] = number[:, 2] * 1.3e-15 * 1e12
    mass[:, 6] = number[:, 3] * 1.6e-15 * 1e12
    mass[:, 7] = number[:, 4] * 1.9e-15 * 1e12   # BC KI
    mass[:, 8] = number[:, 1] * 2e-15 * 1e12
    mass[:, 9] = number[:, 2] * 2.3e-15 * 1e12
    mass[:, 10] = number[:, 3] * 2.6e-15 * 1e12
    mass[:, 11] = number[:, 4] * 2.9e-15 * 1e12  # OC KI
    mass[:, 16] = number[:, 5] * 4.6e-15 * 1e12  # DU AI
    mass[:, 17] = number[:, 6] * 4.9e-15 * 1e12  # DU CI

    crt = np.asarray(_coat(radius))  # (n, 3): KI, AI, CI
    insoluble_sulfate = crt * np.asarray(monolayers)[:, None] * number[:, [4, 5, 6]]

    return dict(number=number, radius=radius, density=density, mass=mass,
                mass_delta=np.zeros((n, 18)), dry_radius=radius[:, :4].copy(),
                insoluble_sulfate=insoluble_sulfate, temperature=np.full(n, 280.),
                pressure=np.full(n, 101325.), timestep=np.full(n, 600.),
                nucleated_number=np.zeros(n), coagulation=np.zeros(n))


MONOLAYERS = [0.0, 0.3, 0.9999, 1.0, 1.0001, 2.0, 5.0]


@pytest.mark.parametrize("compiled", [False, True])
def test_coat_monolayer_thresholds_native(native, compiled):
    data = cases(MONOLAYERS)
    props, _ = run_oracle(native, data, operation="properties")
    mean_mass = props["mean_particle_mass"]
    expected, _ = run_oracle(native, data, operation="dnum")

    function = dnum_all
    if compiled:
        function = jax.jit(function, static_argnames=("coagulation",))
    n = len(MONOLAYERS)
    results = [function(data["mass"][i], data["number"][i], data["temperature"][i], data["pressure"][i],
                         data["radius"][i], data["density"][i], data["insoluble_sulfate"][i],
                         data["nucleated_number"][i], data["mass_delta"][i], mean_mass[i], data["timestep"][i],
                         coagulation=False)
               for i in range(n)]
    mass = np.array([np.asarray(r[0]) for r in results])
    number = np.array([np.asarray(r[1]) for r in results])

    # atol covers a handful of ~1e-18 compiled-path cancellation residuals
    # (mo_ham_m7ctl.F90-scale BC/OC masses, ~1e-1, against float64 eps).
    np.testing.assert_allclose(mass, expected["mass"], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(number, expected["number"], rtol=1e-12, atol=1e-16)

    # Below one monolayer, nothing ages (zcrit_x = min(n_insoluble, zso4/crt)
    # with zso4 < crt => 0 < zcrit, but here zso4 is an exact multiple of
    # crt below 1 so no FULL particle crosses the "min(available, needed)"
    # test: at monolayers<1 some fractional transfer still occurs per the
    # analytical zcrit formula -- only the monolayers==0 cell is an exact
    # no-op. Assert that regime directly:
    np.testing.assert_array_equal(number[0], data["number"][0])
    np.testing.assert_array_equal(mass[0, [7, 11, 16, 17]], data["mass"][0, [7, 11, 16, 17]])
    # Above several monolayers, ageing saturates at "all insoluble number
    # transferred" (zcrit_x clipped to the insoluble number itself).
    np.testing.assert_allclose(number[-1, 4], 0., atol=1e-10)
    np.testing.assert_allclose(number[-1, 5], 0., atol=1e-10)
    np.testing.assert_allclose(number[-1, 6], 0., atol=1e-10)


def test_coat_empty_insoluble_modes_are_untouched(native):
    """Zero insoluble number with nonzero zso4 must not divide by zero or

    move anything (m7_concoag's cmin_aernl/zeps gates, mo_ham_m7.F90:3732-
    3746): starved coating demand with nothing to coat is a no-op, not NaN.
    """
    data = cases([2.0])
    data["number"][0, 4:] = 0.0
    props, _ = run_oracle(native, data, operation="properties")
    mean_mass = props["mean_particle_mass"]
    expected, _ = run_oracle(native, data, operation="dnum")
    mass, number, _ = dnum_all(data["mass"][0], data["number"][0], data["temperature"][0], data["pressure"][0],
                                data["radius"][0], data["density"][0], data["insoluble_sulfate"][0],
                                data["nucleated_number"][0], data["mass_delta"][0], mean_mass[0],
                                data["timestep"][0], coagulation=False)
    np.testing.assert_allclose(np.asarray(mass), expected["mass"][0], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(number), expected["number"][0], rtol=1e-12, atol=1e-18)
    assert np.isfinite(np.asarray(mass)).all()
    np.testing.assert_array_equal(np.asarray(number)[4:], 0.)


def test_coat_zero_radius_insoluble_mode_is_finite():
    """A zero/near-zero radius gives crtcst==0 (m7_coat's surface term), the

    coating gate (`crtcst>zeps`) must exclude it rather than divide by zero.
    """
    radius = np.array([2e-7, 2e-6, 1e-5, 1e-4, 0., 0., 0.])
    out = _coat(radius)
    assert np.isfinite(np.asarray(out)).all()
    np.testing.assert_array_equal(np.asarray(out), 0.)
