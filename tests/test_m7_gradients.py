# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Float64 derivatives are correct, not merely finite.

The default float64 path takes some gradients from a different expression
than the value (``_ad.with_gradient_of`` in condensation, ``_ad.div`` /
``_ad.sqrt`` / ``_ad.cbrt_volume`` custom JVPs everywhere). A wrong rule
there would still give finite gradients. tests/test_m7_float32_precision.py
checks reverse mode through the whole ``step_all`` along one direction, with
the native and the cancellation-free forms; these tests check each rule and
decay branch on its own, and ``cancellation_free=None`` (native value,
cancellation-free derivative), comparing forward and reverse mode against
central finite differences in float64.

Inputs are perturbed relatively (``x*exp(u)`` around ``u = 0``) so that one
step size suits quantities from 1e-12 to 1e10, and the designed points sit
away from every branch threshold so the finite difference stays on one
branch.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from m7_jax import _ad
from m7_jax.coagulation import _decay
from m7_jax.condensation import produce_and_condense
from m7_jax.constants import ZMIN_COAG
from m7_jax.model import FullState, step_all
from m7_jax.properties import default_kappa_table_path, load_kappa_table
from tests.m7_case_space import random_cases

jax.config.update('jax_enable_x64', True)


def _relative(fn, base):
    """``fn`` of ``base*exp(u)``, as a function of ``u`` (one per argument)."""
    base = tuple(jnp.asarray(b, dtype=jnp.float64) for b in base)
    return (lambda *u: fn(*(b*jnp.exp(v) for b, v in zip(base, u))),
            tuple(jnp.zeros_like(b) for b in base))


# --- The custom JVP rules themselves. -----------------------------------------

@pytest.mark.parametrize('rule, base', [
    (_ad.div, (np.array([3., -2e-7, 5e12]), np.array([7e-3, 4e5, -1.5]))),
    (_ad.sqrt, (np.array([2e-20, .3, 4e9]),)),
    (_ad.cbrt_volume, (np.array([1e-21, .3, 4e9]),)),
])
def test_custom_jvp_rules_match_finite_differences(rule, base):
    fn, u = _relative(rule, base)
    check_grads(fn, u, order=2, modes=('fwd', 'rev'), eps=1e-4, rtol=1e-6)


def test_masked_zero_derivatives_are_finite():
    """At 0 the rules return a finite derivative (cbrt_volume 0; sqrt 1, from
    its ``where(zero, x, ...)`` value branch), so a masked branch's zero
    cotangent gives a zero gradient, not NaN as plain ``sqrt``/``**(1/3)`` do."""
    for rule, plain in ((_ad.sqrt, jnp.sqrt), (_ad.cbrt_volume, lambda x: x**(1/3))):
        assert np.isfinite(float(jax.grad(rule)(0.)))
        masked = lambda f: (lambda x: jnp.where(x > 0., f(x), 0.))
        assert float(jax.grad(masked(rule))(0.)) == 0.
        assert np.isnan(float(jax.grad(masked(plain))(0.)))


# --- The analytical number decay (m7_delcoa), every branch. -------------------

# (initial, formed, a, b) per branch; timestep 600 s. ``formed`` sets the
# source formed/dt (>= 1 selects the nucleation branches); b against
# ZMIN_COAG selects inter-modal loss. Values are chosen so that neither the
# MIN(., total) clamp nor the bottom < ZMIN_COAG fallback is near.
DECAY_BRANCHES = {
    'intra-only': (2e4, 0., 3e-10, 0.),
    'intra+inter': (2e4, 0., 3e-10, 4e-5),
    'source+inter': (3e3, 6e4, 2e-9, 8e-5),
    'source+intra': (3e3, 6e4, 2e-9, 0.),
}


@pytest.mark.parametrize('robust', [False, True])
@pytest.mark.parametrize('branch', sorted(DECAY_BRANCHES))
def test_analytical_decay_gradients_match_finite_differences(branch, robust):
    initial, formed, a, b = DECAY_BRANCHES[branch]
    assert (b >= ZMIN_COAG) == ('inter' in branch)

    def decay(initial, formed, a, b):
        new, intra, inter = _decay(initial + formed, initial, a, b, formed, 600., robust=robust)
        return jnp.stack([new, intra, inter])

    # Relative perturbation of a zero is no perturbation; keep zeros fixed.
    names = ('initial', 'formed', 'a', 'b')
    values = dict(zip(names, (initial, formed, a, b)))
    free = [name for name in names if values[name] != 0.]

    def of_free(*free_values):
        return decay(**{**values, **dict(zip(free, free_values))})

    fn, u = _relative(of_free, [values[name] for name in free])
    check_grads(fn, u, order=1, modes=('fwd', 'rev'), eps=1e-5, rtol=1e-6)


# --- Condensation: native value, cancellation-free derivative. -----------------

@pytest.mark.parametrize('choice', [None, True])
@pytest.mark.parametrize('cloud', [0., .4])
def test_condensation_gradients_match_finite_differences(choice, cloud):
    """``cancellation_free=None`` returns the native value and differentiates
    the cancellation-free form through ``_ad.with_gradient_of``; both must
    give the derivative of the same function."""
    sulfate = np.array([1e5, 2e6, 3e7, 4e6])
    modal_sink = np.array([2e-4, 1e-4, 5e-4, 3e-5, 1e-6, 2e-6, 0.])

    def condense(gas, production, sink):
        result = produce_and_condense(gas, production, sulfate, sink, modal_sink, cloud, 900.,
                                      cancellation_free=choice)
        return jnp.concatenate([result.gas[None], result.sulfate, result.insoluble_sulfate])

    for gas, production, sink in ((3e7, 2e4, 8.32e-4), (1e5, 5e5, 2e-3), (5e8, 10., 1e-5)):
        fn, u = _relative(condense, (gas, production, sink))
        check_grads(fn, u, order=1, modes=('fwd', 'rev'), eps=1e-5, rtol=1e-6)


# --- The whole step_all. ------------------------------------------------------

@pytest.fixture(scope='module')
def kappa_table():
    return load_kappa_table(default_kappa_table_path())


@pytest.mark.parametrize('choice', [None, True])
def test_step_all_gradients_match_finite_differences(kappa_table, choice):
    """A scalar per-cell objective of the full step, nucleation schemes 1
    plus organic 1, differentiated along a random relative direction in
    mass, number, gas and (additively) temperature and humidity."""
    n = 48
    cases = random_cases(n, 11)
    fixed = {k: jnp.asarray(cases[k]) for k in ('pressure', 'timestep', 'production', 'cloud_fraction',
                                                'forest_fraction', 'in_pbl')}

    def outputs(mass, number, gas, temperature, humidity):
        result = step_all(FullState(mass, number, gas), temperature, fixed['pressure'], humidity,
                          fixed['timestep'], fixed['production'], fixed['cloud_fraction'], kappa_table,
                          fixed['forest_fraction'], fixed['in_pbl'], nucleation_scheme=1, organic_scheme=1,
                          cancellation_free=choice)
        return result.state

    base = tuple(jnp.asarray(cases[k]) for k in ('mass', 'number', 'gas', 'temperature',
                                                 'relative_humidity'))
    # Fixed positive weights, normalised by the unperturbed outputs, make the
    # objective a dimensionless O(1) sum per cell.
    rng = np.random.default_rng(3)
    reference = outputs(*base)
    # Per-component normalisation would hand a near-empty component (1e-20
    # of its cell, or a gas consumed to 0) a huge weight and turn its
    # rounding into the objective, so each field is scaled by its per-cell
    # total, input plus output.
    def cell_total(field):
        field = np.abs(np.asarray(field))
        return np.sum(np.reshape(field, (n, -1)), -1).reshape((n,) + (1,)*(field.ndim - 1))

    weights = [rng.uniform(.5, 1.5, np.shape(r))/(cell_total(r) + cell_total(b) + 1e-30)
               for r, b in zip(reference, base[:3])]

    def objective(t):
        """Per-cell objective at ``base`` moved by ``t`` along the direction."""
        mass, number, gas, temperature, humidity = base
        state = outputs(mass*jnp.exp(t[:, None]*direction[0]), number*jnp.exp(t[:, None]*direction[1]),
                        gas*jnp.exp(t*direction[2]), temperature + t*direction[3],
                        humidity + t*direction[4])
        return sum(jnp.sum(jnp.reshape(s*w, (n, -1)), -1) for s, w in zip(state, weights))

    direction = (rng.normal(size=(n, 18)), rng.normal(size=(n, 7)), rng.normal(size=n),
                 rng.normal(size=n), 1e-2*rng.normal(size=n))
    zero = jnp.zeros(n)
    forward = jax.jvp(objective, (zero,), (jnp.ones(n),))[1]
    reverse = jax.grad(lambda t: jnp.sum(objective(t)))(zero)

    def central(h):
        return (objective(zero + h) - objective(zero - h))/(2*h)

    coarse, fine = np.asarray(central(1e-4)), np.asarray(central(5e-5))
    scale = np.abs(fine) + 1.
    # A cell whose finite difference changes with the step size straddles a
    # branch (a gate, a MIN/MAX, a lookup-table node) and has no derivative
    # to compare against.
    smooth = np.abs(coarse - fine) <= 1e-6*scale
    assert smooth.sum() >= 0.9*n, int(smooth.sum())
    # Forward and reverse mode agree to rounding: measured 1.1e-11 absolute
    # on an O(1) objective.
    np.testing.assert_allclose(np.asarray(forward), np.asarray(reverse), rtol=1e-9, atol=1e-10)
    # Measured on all 48 cells (all smooth at these steps): max error 2.1e-7
    # with cancellation_free=None (the native value's own cancellation noise
    # in the finite difference) and 9.2e-9 with True.
    error = np.abs(np.asarray(reverse) - fine)/scale
    assert np.max(error[smooth]) <= 1e-6, (np.max(error[smooth]), np.argmax(np.where(smooth, error, 0.)))
