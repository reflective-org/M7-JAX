"""Float32 precision adaptations (``m7_jax.precision.cancellation_free``).

Every site where a native M7 expression obtains a small net quantity as the
difference of two nearly equal totals has a cancellation-free form that is
identical in real arithmetic and is selected in any dtype narrower than
float64. These tests check, per site:

- the cancellation-free form equals a 50-digit evaluation of the *native*
  expression (so the two forms really are the same mathematics) in float64
  and float32, in regimes where the native float32 form is pure rounding
  noise or of the wrong sign;
- float64 is untouched: the native form is still the one selected there (the
  parity suite against the compiled r7492 Fortran guards its bits).

Like ``test_m7_step_all_float32.py`` this module does not touch
``jax_enable_x64`` at import time; each test sets the precision it needs and
restores the previous setting.
"""
import contextlib
import functools
from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from m7_jax import precision
from m7_jax.coagulation import _decay, _mixed_particles, dnum_all
from m7_jax.condensation import produce_and_condense
from m7_jax.properties import (average_properties, default_kappa_table_path, kappa_hydration,
    load_kappa_table)

getcontext().prec = 50


@contextlib.contextmanager
def x64(enabled):
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update('jax_enable_x64', enabled)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def D(value):
    return Decimal(float(value))


def _rel(actual, exact):
    return abs(D(actual) - exact) / abs(exact)


def test_float64_keeps_native_forms_and_narrower_dtypes_do_not():
    assert not precision.cancellation_free(jnp.float64)
    assert precision.cancellation_free(jnp.float32)
    assert precision.cancellation_free(jnp.bfloat16)


# --- The review case: KS sulfate and number went negative in float32. -------

def _review_case(dtype):
    mass = jnp.zeros((1, 18), dtype).at[0, 0].set(1.6678106e8).at[0, 7].set(.010660719)
    number = jnp.zeros((1, 7), dtype).at[0, 0].set(440305.4).at[0, 4].set(52.36747)
    avg = average_properties(mass, number)
    table = load_kappa_table(default_kappa_table_path())
    table = table._replace(**{k: jnp.asarray(getattr(table, k), dtype) for k in table._fields})
    hyd = kappa_hydration(avg.pttn, number, 220., .4, table, previous=avg)
    return (mass, number, 220., 25000., hyd.wet_radius, hyd.density, jnp.zeros((1, 3), dtype),
            jnp.zeros(1, dtype), jnp.zeros_like(mass), avg.pttn, .37276176)


@pytest.mark.parametrize('compiled', [False, True])
def test_review_case_float32_coagulation_is_nonnegative_and_accurate(compiled):
    """4.4e5 NS particles on 52 KI at 220 K: the native float32 2.2.3 mixed-
    particle count was negative and returned KS sulfate -1022 molecules cm-3
    and KS number -0.029 cm-3. The float64 result is 199.13 and 0.0055973."""
    with x64(True):
        mass64, number64, _ = dnum_all(*_review_case(jnp.float64))
        mass64, number64 = np.asarray(mass64), np.asarray(number64)
    with x64(False):
        function = jax.jit(dnum_all) if compiled else dnum_all
        mass, number, delta = (np.asarray(x) for x in function(*_review_case(jnp.float32)))
    assert mass.dtype == np.float32
    assert np.all(np.isfinite(mass)) and np.all(np.isfinite(number)) and np.all(np.isfinite(delta))
    assert np.all(mass >= 0) and np.all(number >= 0) and np.all(delta >= 0)
    np.testing.assert_allclose(mass[0, 1], mass64[0, 1], rtol=1e-5)
    np.testing.assert_allclose(number[0, 1], number64[0, 1], rtol=1e-5)
    np.testing.assert_allclose(mass, mass64, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(number, number64, rtol=1e-5, atol=1e-12)


# --- m7_delcoa 2.2.3: zanytnm. ---------------------------------------------

def _zanytnm_exact(zaerni, zaerns, a0dt):
    i, s, c = D(zaerni), D(zaerns), D(a0dt)
    t = s + i
    den = 2 + c * t
    zi = 4 * i / (den * (2 + c * (t - i))) if i > 1 else Decimal(0)
    value = 2 * t / den - (zi + 4 * s / (den * (2 + c * (t - s))))
    return min(value, i)


# (zaerni, zaerns, a*dt): the review case, a tiny insoluble population (the
# I <= 1 branch, where float64's native form already loses 4 digits), and
# well-conditioned designed values.
ZANYTNM_CASES = [(52.36746861, 440305.40625, 1.30964694e-10), (1.4831929e-10, 5464.375, 1.1309107e-06),
                 (2.5227053e-10, 1102.0138, 1.0315359e-07), (5.5, 2.03e4, 1.7e-6), (2050., 8300., 3e-6),
                 (0.3, 1e3, 5e-4), (40., 41., 2e-2)]


@pytest.mark.parametrize('zaerni,zaerns,a0dt', ZANYTNM_CASES)
@pytest.mark.parametrize('dtype,rtol', [(jnp.float32, 2e-6), (jnp.float64, 5e-15)])
def test_mixed_particles_cancellation_free_form_equals_native_expression(zaerni, zaerns, a0dt, dtype, rtol):
    exact = _zanytnm_exact(zaerni, zaerns, a0dt)
    with x64(dtype == jnp.float64):
        cast = functools.partial(jnp.asarray, dtype=dtype)
        value = _mixed_particles(cast(zaerni), cast(zaerns), cast(zaerni) + cast(zaerns), cast(a0dt), robust=True)
        assert float(value) >= 0
        assert _rel(value, exact) < rtol


def test_native_mixed_particles_in_float32_is_negative_on_the_review_case():
    """Documents the defect the robust form removes (and that the test above
    is meaningful): the native float32 expression is negative here."""
    zaerni, zaerns, a0dt = ZANYTNM_CASES[0]
    with x64(False):
        cast = functools.partial(jnp.asarray, dtype=jnp.float32)
        native = _mixed_particles(cast(zaerni), cast(zaerns), cast(zaerni) + cast(zaerns), cast(a0dt), robust=False)
    assert float(native) < 0 < _zanytnm_exact(zaerni, zaerns, a0dt)


# --- m7_delcoa 1.2/2.2: the analytical number solution and its loss. --------

def _decay_exact(total, initial, a, b, formed, dt):
    """50-digit evaluation of the native branch formulas (mo_ham_m7.F90:
    4101-4366), returning (new, intra_loss, inter_loss)."""
    n, n0, a, b, dt = D(total), D(initial), D(a), D(b), D(dt)
    c = D(formed) / dt
    inter = b >= D(np.float32(1e-15))
    if c < 1:
        if inter:
            e = (-b * dt).exp()
            new = min(b * n * e / (n * a * (1 - e) + b), n)
        else:
            new = n / (1 + n * a * dt)
    elif inter:
        r = (b * b + 4 * a * c).sqrt()
        f2 = 2 * a * n0 + b
        f3 = (-r * dt).exp() * (r - f2) / (r + f2)
        new = min((r * (1 - f3) / (1 + f3) - b) / 2 / a, n)
    else:
        r1 = (c / a).sqrt()
        e = (-r1 * 2 * a * dt).exp()
        ratio = (n0 + r1) / (n0 - r1)
        new = n if ratio - e < D(np.float32(1e-15)) else min(r1 * (ratio + e) / (ratio - e), n)
    loss = n - new
    if not inter:
        return new, loss, Decimal(0)
    intra, inter_rate = n * n * a, n * b
    return new, loss * intra / (intra + inter_rate), loss * inter_rate / (intra + inter_rate)


# (total, initial, a, b, formed, dt), formed = 0 unless nucleating:
DECAY_CASES = [
    (440305.4, 440305.4, 3.5e-10, 1.6e-9, 0., 720.),        # review case NS: tiny b*dt
    (2e4, 2e4, 4e-10, 3e-4, 0., 600.),                      # ordinary inter+intra
    (52.4, 52.4, 1.1e-9, 0., 0., 720.),                     # intra only, little loss
    (5e5, 5e5, 3e-10, 0., 0., 1800.),                       # intra only, strong
    (3e4, 0., 1e-9, 2e-3, 3e4, 720.),                       # nucleation into empty NS, strongly scavenged
    (3.1e4, 1e3, 5e-10, 1e-2, 3e4, 600.),                   # nucleation, 2a*n << b
    (2.5e5, 2e5, 1e-9, 1e-5, 5e4, 1200.),                   # nucleation, weak inter
    (3.02e4, 2e4, 1e-9, 0., 1e4, 600.),                     # nucleation, intra only, n0 > R1
]


@pytest.mark.parametrize('case', DECAY_CASES)
@pytest.mark.parametrize('dtype,rtol', [(jnp.float32, 3e-5), (jnp.float64, 1e-12)])
def test_decay_cancellation_free_forms_equal_native_expressions(case, dtype, rtol):
    exact = _decay_exact(*case)
    with x64(dtype == jnp.float64):
        values = _decay(*(jnp.asarray(x, dtype=dtype) for x in case), robust=True)
    for value, reference in zip(values, exact):
        assert float(value) >= 0
        if reference == 0:
            assert float(value) == 0
        else:
            # With a source, zatot = zansum - zaernt stays a difference
            # (see _decay); its error is one rounding of zansum.
            scale = abs(reference) if case[4] == 0 else max(abs(reference), D(case[0]) * D(6e-8 if dtype == jnp.float32 else 1e-16))
            assert abs(D(value) - reference) / scale < rtol, (value, reference)


# --- m7_concoag: ageing every insoluble particle. ---------------------------

@pytest.mark.parametrize('coagulation', [False, True])
def test_full_ageing_leaves_no_negative_insoluble_mass_in_float32(coagulation):
    """With far more condensed sulfate than the coating needs, every KI/AI/CI
    particle ages (zcrit = number). The native ``mass - zcrit*pttn*1e12``
    leaves rounding noise of either sign (in float64 too); the factored form
    leaves exactly zero, or, where KI coagulated, the intra-modal mass that
    section 1 keeps without a particle (``zansq*pttn*1e12``, non-negative)."""
    rng = np.random.default_rng(7)
    n = 256
    number = np.tile([2e4, 300., 30., .2, 0., 0., 0.], (n, 1))
    number[:, 4:] = np.exp(rng.uniform(np.log(1e-3), np.log(1e4), (n, 3)))
    mass = np.zeros((n, 18))
    mass[:, :4] = [2e6, 2e8, 3e9, 4e9]
    for column, mode in ((7, 4), (11, 4), (16, 5), (17, 6)):
        mass[:, column] = number[:, mode] * np.exp(rng.uniform(np.log(1e-16), np.log(1e-13), n)) * 1e12
    with x64(False):
        f32 = functools.partial(jnp.asarray, dtype=jnp.float32)
        radius = f32(np.tile([2e-7, 2e-6, 1e-5, 1e-4, 3e-6, 2e-5, 1.5e-4], (n, 1)))
        density = f32(np.tile([1.841, 1.7, 1.4, 1.2, 1.8, 2.6, 2.6], (n, 1)))
        avg = average_properties(f32(mass), f32(number))
        result = dnum_all(f32(mass), f32(number), 250., 5e4, radius, density, f32(np.full((n, 3), 1e14)),
                          f32(np.zeros(n)), f32(np.zeros((n, 18))), avg.pttn, 600., coagulation=coagulation)
        out_mass, out_number = np.asarray(result[0]), np.asarray(result[1])
    assert np.all(out_mass >= 0) and np.all(out_number >= 0)
    np.testing.assert_array_equal(out_number[:, 4:], 0.)
    if not coagulation:
        np.testing.assert_array_equal(out_mass[:, [7, 11, 16, 17]], 0.)


# --- m7_prod_cond. -----------------------------------------------------------

def _condense_exact(gas, production, sink, cloud, dt):
    g, p, k, c, t = D(gas), D(production), D(sink), D(cloud), D(dt)
    e = (-k * t).exp()
    final = max((g - p / k) * e + p / k, Decimal(0))
    # Mode 2 is the only one with a sink, so it also takes the cloudy part.
    return (1 - c) * final, (1 - c) * (g - final + p * t) + c * (g + p * t)


# (gas, production, sink, cloud, dt): a large reservoir that barely condenses,
# production-dominated near steady state, a tiny sink, and ordinary cases.
CONDENSE_CASES = [(1.37e8, 6.5e4, 2.2e-9, .93, 140.), (1e8, 1e5, 3e-3, 0., 720.), (1e3, 5e4, 1e-6, .3, 600.),
                  (1768., 1005., 2e-9, .42, 664.), (5e7, 2e4, 1e-2, .1, 1800.), (1e6, 0., 4e-4, 0., 720.)]


@pytest.mark.parametrize('case', CONDENSE_CASES)
@pytest.mark.parametrize('dtype,rtol', [(jnp.float32, 2e-5), (jnp.float64, 1e-13)])
def test_condensation_cancellation_free_forms_equal_native_expressions(case, dtype, rtol):
    gas, production, sink, cloud, dt = case
    final_exact, condensed_exact = _condense_exact(*case)
    modal = np.array([0., 0., sink, 0., 0., 0., 0.])
    with x64(dtype == jnp.float64), _forced(True):
        cast = functools.partial(jnp.asarray, dtype=dtype)
        result = produce_and_condense(cast(gas), cast(production), jnp.zeros(4, dtype), cast(sink), cast(modal),
                                      cast(cloud), cast(dt))
    assert float(result.gas) >= 0 and np.all(np.asarray(result.sulfate) >= 0)
    assert _rel(result.gas, final_exact) < rtol
    # x + expm1(-x) keeps an absolute error of one rounding of P*dt.
    floor = D(production) * D(dt) * D(1e-7 if dtype == jnp.float32 else 1e-16)
    assert abs(D(result.sulfate[2]) - condensed_exact) / max(abs(condensed_exact), floor) < rtol


@contextlib.contextmanager
def _forced(value):
    """Select the cancellation-free forms regardless of dtype (eager only:
    the switch is read at trace time)."""
    original = precision.cancellation_free
    precision.cancellation_free = lambda dtype: value or original(dtype)
    try:
        yield
    finally:
        precision.cancellation_free = original
