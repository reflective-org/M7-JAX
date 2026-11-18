"""Parity, chunking, differentiability, and dtype tests for ``step_all_soa`` and SoA helpers."""
from contextlib import contextmanager
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from m7_jax.interface import (
    aerosol_mass_to_native,
    aerosol_mass_to_native_soa,
    native_to_aerosol_mass,
    native_to_aerosol_mass_soa,
    native_to_number_mixing_ratio,
    native_to_number_mixing_ratio_soa,
    number_mixing_ratio_to_native,
    number_mixing_ratio_to_native_soa,
)
from m7_jax.model import FULL_STATIC_OPTIONS, FullState, SOA_STATIC_OPTIONS, SoAState, step_all, step_all_soa
from m7_jax.nucleation import KazilLovejoyTable, PreparedKazilLovejoyTable, prepare_kazil_lovejoy_table
from m7_jax.properties import PreparedKappaTable, load_kappa_table, prepare_kappa_table
from tools.m7_oracle import ROOT


@contextmanager
def _x64_mode(enabled: bool):
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update('jax_enable_x64', enabled)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', prev)


def _sample_grid_cases(dtype=np.float64):
    """Build a (2, 7) = 14-cell grid covering diverse M7 aerosol/gas/insoluble regimes."""
    n = 7
    mass = np.zeros((n, 18), dtype=np.float64)
    number = np.zeros((n, 7), dtype=np.float64)
    mass[:, 0:4] = [
        [1e6, 5e8, 5e10, 1e10],
        [5e6, 2e9, 3e11, 2e10],
        [1e5, 1e8, 2e10, 3e10],
        [1e7, 1e9, 1e10, 0.0],
        [1e6, 1e9, 1e11, 1e10],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1e8, 2e10, 5e10],
    ]
    mass[1, 4:8] = [0.05, 0.4, 0.02, 0.8]
    mass[1, 8:12] = [0.2, 2.0, 0.1, 1.5]
    mass[3, 7], mass[3, 11] = 0.01, 0.02
    mass[4, 5], mass[4, 9] = 0.1, 0.5
    mass[0, 12:14] = [1.0, 20.0]
    mass[6, 12:14] = [3.0, 60.0]
    mass[2, 14:18] = [2.0, 30.0, 15.0, 200.0]
    mass[4, 16:18] = [0.5, 3.0]
    number[:] = [
        [800, 300, 100, 1, 0, 0, 0],
        [5000, 3000, 1500, 2, 2000, 0, 0],
        [100, 50, 80, 5, 0, 3, 1],
        [20000, 800, 30, 0, 50, 0, 0],
        [1000, 900, 400, 2, 0, 0.5, 0.2],
        [0, 0, 0, 0, 0, 0, 0],
        [50, 40, 60, 8, 0, 0, 0],
    ]
    # Tile into (2, 7) with perturbed second level so every cell is distinct.
    mass2 = np.stack([mass, mass * 1.15 + 1e-4], axis=0).astype(dtype)
    num2 = np.stack([number, number * 0.9 + 0.25], axis=0).astype(dtype)
    gas = np.stack([
        np.array([1e6, 5e7, 1e6, 3e7, 2e7, 1e8, 1e5]),
        np.array([2e6, 3e7, 5e6, 4e7, 1e7, 5e7, 8e5]),
    ], axis=0).astype(dtype)
    temp = np.stack([
        np.array([290.0, 285.0, 300.0, 220.0, 275.0, 260.0, 295.0]),
        np.array([280.0, 270.0, 290.0, 215.0, 268.0, 255.0, 288.0]),
    ], axis=0).astype(dtype)
    press = np.stack([
        np.array([101325.0, 95000.0, 85000.0, 25000.0, 80000.0, 50000.0, 100000.0]),
        np.array([90000.0, 80000.0, 75000.0, 22000.0, 70000.0, 45000.0, 92000.0]),
    ], axis=0).astype(dtype)
    rh = np.stack([
        np.array([0.85, 0.6, 0.25, 0.4, 0.95, 0.5, 0.9]),
        np.array([0.75, 0.5, 0.35, 0.45, 0.88, 0.6, 0.82]),
    ], axis=0).astype(dtype)
    prod = np.stack([
        np.array([1e3, 5e4, 1e3, 2e4, 1e4, 1e5, 0.0]),
        np.array([2e3, 3e4, 5e2, 1.5e4, 8e3, 6e4, 1e2]),
    ], axis=0).astype(dtype)
    cloud = np.stack([
        np.array([0.0, 0.2, 0.0, 0.0, 0.7, 0.0, 0.3]),
        np.array([0.1, 0.0, 0.15, 0.05, 0.5, 0.0, 0.2]),
    ], axis=0).astype(dtype)
    forest = np.broadcast_to(np.array([0.0, 0.4, 0.0, 0.0, 0.8, 1.0, 0.0], dtype=dtype), (2, 7)).copy()
    in_pbl = np.array([[True, True, False, False, True, True, True],
                       [False, False, False, False, True, False, True]])
    ipr = np.full((2, 7), 5.0, dtype=dtype)
    return dict(
        mass=mass2, number=num2, gas=gas, temperature=temp, pressure=press,
        humidity=rh, timestep=dtype(720.0), production=prod, cloud=cloud,
        forest=forest, in_pbl=in_pbl, ipr=ipr,
    )


def _synthetic_kl_table(dtype=jnp.float64):
    t = jnp.linspace(190.0, 300.0, 4, dtype=dtype)
    rh = jnp.linspace(5.0, 95.0, 4, dtype=dtype)
    h2so4 = jnp.array([1e5, 1e6, 1e7, 1e8, 1e9], dtype=dtype)
    ipr = jnp.array([0.5, 2.0, 10.0, 30.0], dtype=dtype)
    cs = jnp.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=dtype)
    grid = (
        -3.0
        + 0.01 * (t[:, None, None, None, None] - 240.0)
        + 0.02 * rh[None, :, None, None, None]
        + 0.5 * jnp.log10(h2so4)[None, None, :, None, None]
        + 0.1 * jnp.log10(ipr)[None, None, None, :, None]
        - 0.2 * jnp.log10(cs)[None, None, None, None, :]
    )
    return KazilLovejoyTable(t, rh, h2so4, ipr, cs, grid.astype(dtype))


@pytest.mark.parametrize('scheme,organic,coag,cond', [
    (0, 0, True, True),
    (1, 0, True, True),
    (1, 2, True, True),
    (2, 1, True, True),
    (1, 0, False, True),
    (0, 0, True, False),
])
def test_step_all_soa_bit_identical_to_step_all(scheme, organic, coag, cond):
    with _x64_mode(True):
        table = load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')
        kl = _synthetic_kl_table(jnp.float64) if scheme == 2 else None
        d = _sample_grid_cases(np.float64)

        aos_fn = jax.jit(functools.partial(
            step_all,
            coagulation=coag, condensation=cond,
            nucleation_scheme=scheme, organic_scheme=organic,
        ))
        soa_fn = jax.jit(functools.partial(
            step_all_soa,
            coagulation=coag, condensation=cond,
            nucleation_scheme=scheme, organic_scheme=organic,
        ))

        res_aos = aos_fn(
            FullState(d['mass'], d['number'], d['gas']),
            d['temperature'], d['pressure'], d['humidity'], d['timestep'],
            d['production'], d['cloud'], table, d['forest'], d['in_pbl'],
            ion_pair_rate=d['ipr'], kazil_lovejoy_table=kl,
        )
        soa_state = SoAState(
            tuple(d['mass'][..., i] for i in range(18)),
            tuple(d['number'][..., i] for i in range(7)),
            d['gas'],
        )
        res_soa = soa_fn(
            soa_state,
            d['temperature'], d['pressure'], d['humidity'], d['timestep'],
            d['production'], d['cloud'], prepare_kappa_table(table), d['forest'], d['in_pbl'],
            ion_pair_rate=d['ipr'], kazil_lovejoy_table=kl,
        )

        np.testing.assert_array_equal(np.stack(res_soa.state.mass, axis=-1), np.asarray(res_aos.state.mass))
        np.testing.assert_array_equal(np.stack(res_soa.state.number, axis=-1), np.asarray(res_aos.state.number))
        np.testing.assert_array_equal(np.asarray(res_soa.state.gas), np.asarray(res_aos.state.gas))
        np.testing.assert_array_equal(
            np.stack(res_soa.properties.dry_radius, axis=-1), np.asarray(res_aos.properties.dry_radius)
        )
        np.testing.assert_array_equal(
            np.stack(res_soa.properties.wet_radius, axis=-1), np.asarray(res_aos.properties.wet_radius)
        )
        np.testing.assert_array_equal(
            np.stack(res_soa.properties.density, axis=-1), np.asarray(res_aos.properties.density)
        )
        np.testing.assert_array_equal(
            np.stack(res_soa.properties.water, axis=-1), np.asarray(res_aos.properties.water)
        )
        np.testing.assert_array_equal(np.asarray(res_soa.valid), np.asarray(res_aos.valid))


@pytest.mark.parametrize('chunk_size,remat_chunks', [
    (3, True),
    (5, False),
    (7, True),
    (32, True),
])
def test_step_all_soa_chunking_matches_unchunked_and_preserves_gradients(chunk_size, remat_chunks):
    with _x64_mode(True):
        table = prepare_kappa_table(load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc'))
        d = _sample_grid_cases(np.float64)
        soa_state = SoAState(
            tuple(jnp.asarray(d['mass'][..., i]) for i in range(18)),
            tuple(jnp.asarray(d['number'][..., i]) for i in range(7)),
            jnp.asarray(d['gas']),
        )
        base_fn = jax.jit(functools.partial(
            step_all_soa, nucleation_scheme=1, organic_scheme=1,
            compute_valid=False, compute_water=False, chunk_size=None,
        ))
        chunk_fn = jax.jit(functools.partial(
            step_all_soa, nucleation_scheme=1, organic_scheme=1,
            compute_valid=False, compute_water=False,
            chunk_size=chunk_size, remat_chunks=remat_chunks,
        ))

        out_base = base_fn(
            soa_state, d['temperature'], d['pressure'], d['humidity'], d['timestep'],
            d['production'], d['cloud'], table, d['forest'], d['in_pbl'],
        )
        out_chunk = chunk_fn(
            soa_state, d['temperature'], d['pressure'], d['humidity'], d['timestep'],
            d['production'], d['cloud'], table, d['forest'], d['in_pbl'],
        )
        assert out_chunk.valid is None
        assert out_chunk.properties.water == ()
        np.testing.assert_allclose(
            np.stack(out_chunk.state.mass, axis=-1), np.stack(out_base.state.mass, axis=-1), rtol=1e-14, atol=0
        )
        np.testing.assert_allclose(
            np.stack(out_chunk.state.number, axis=-1), np.stack(out_base.state.number, axis=-1), rtol=1e-14, atol=0
        )
        np.testing.assert_allclose(np.asarray(out_chunk.state.gas), np.asarray(out_base.state.gas), rtol=1e-14, atol=0)

        def loss_fn(csize):
            def _loss(gas_in, m0_in, n0_in):
                st = SoAState((m0_in, *soa_state.mass[1:]), (n0_in, *soa_state.number[1:]), gas_in)
                res = step_all_soa(
                    st, d['temperature'], d['pressure'], d['humidity'], d['timestep'],
                    d['production'], d['cloud'], table, d['forest'], d['in_pbl'],
                    nucleation_scheme=1, organic_scheme=1,
                    compute_valid=False, compute_water=False,
                    chunk_size=csize, remat_chunks=remat_chunks,
                )
                return (
                    jnp.sum(res.state.gas * 1e-6)
                    + jnp.sum(res.state.mass[0] * 1e-7)
                    + jnp.sum(res.state.number[1] * 1e-3)
                    + jnp.sum(res.properties.wet_radius[2] * 1e4)
                )
            return jax.jit(jax.grad(_loss, argnums=(0, 1, 2)))

        g_base = loss_fn(None)(soa_state.gas, soa_state.mass[0], soa_state.number[0])
        g_chunk = loss_fn(chunk_size)(soa_state.gas, soa_state.mass[0], soa_state.number[0])
        for gb, gc in zip(g_base, g_chunk):
            gb_np, gc_np = np.asarray(gb), np.asarray(gc)
            assert bool(jnp.all(jnp.isfinite(gc)))
            # Especially verifies the last real cell is not contaminated by tail padding!
            mask = np.ones_like(gb_np, dtype=bool)
            mask[0, 3] = False  # cold-ut has native ~1e-9 cancellation rounding (see test_m7_step_all.py:132)
            np.testing.assert_allclose(gc_np[mask], gb_np[mask], rtol=1e-12, atol=1e-24)
            np.testing.assert_allclose(gc_np[0, 3], gb_np[0, 3], rtol=5e-9, atol=1e-24)


def test_step_all_chunking_options_are_jit_static():
    """``chunk_size`` picks Python control flow, so both entry points list it as static."""
    with _x64_mode(True):
        table = load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')
        d = _sample_grid_cases(np.float64)
        args = (FullState(d['mass'], d['number'], d['gas']), d['temperature'], d['pressure'],
                d['humidity'], d['timestep'], d['production'], d['cloud'], table, d['forest'], d['in_pbl'])
        call = jax.jit(step_all, static_argnames=FULL_STATIC_OPTIONS)
        base = call(*args, organic_scheme=1)
        chunked = call(*args, organic_scheme=1, chunk_size=5, remat_chunks=False)
        np.testing.assert_allclose(np.asarray(chunked.state.mass), np.asarray(base.state.mass), rtol=1e-14, atol=0)
        np.testing.assert_allclose(np.asarray(chunked.state.number), np.asarray(base.state.number),
                                   rtol=1e-14, atol=0)
        np.testing.assert_allclose(np.asarray(chunked.state.gas), np.asarray(base.state.gas), rtol=1e-14, atol=0)
        assert set(FULL_STATIC_OPTIONS) <= set(SOA_STATIC_OPTIONS)


@pytest.mark.parametrize('x64_enabled', [False, True])
def test_step_all_soa_float32_stays_float32_and_differentiable(x64_enabled):
    with _x64_mode(x64_enabled):
        table64 = load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')
        table32 = prepare_kappa_table(table64, dtype=jnp.float32)
        kl32 = prepare_kazil_lovejoy_table(_synthetic_kl_table(jnp.float32), dtype=jnp.float32)
        assert isinstance(table32, PreparedKappaTable)
        assert isinstance(kl32, PreparedKazilLovejoyTable)
        assert table32.growth_thrk_flat.dtype == jnp.float32
        assert kl32.log_pfr_flat.dtype == jnp.float32
        # Idempotent preparation
        assert prepare_kappa_table(table32, dtype=jnp.float32) is table32
        assert prepare_kazil_lovejoy_table(kl32, dtype=jnp.float32) is kl32

        d = _sample_grid_cases(np.float32)
        soa_state = SoAState(
            tuple(jnp.asarray(d['mass'][..., i], dtype=jnp.float32) for i in range(18)),
            tuple(jnp.asarray(d['number'][..., i], dtype=jnp.float32) for i in range(7)),
            jnp.asarray(d['gas'], dtype=jnp.float32),
        )

        for scheme, kl in ((1, None), (2, kl32)):
            fn = jax.jit(functools.partial(
                step_all_soa, nucleation_scheme=scheme, organic_scheme=1,
                compute_valid=True, compute_water=True, chunk_size=5,
            ))
            out = fn(
                soa_state, d['temperature'], d['pressure'], d['humidity'], jnp.float32(720.0),
                d['production'], d['cloud'], table32, d['forest'], d['in_pbl'],
                ion_pair_rate=d['ipr'], kazil_lovejoy_table=kl,
            )
            assert out.state.gas.dtype == jnp.float32
            assert all(x.dtype == jnp.float32 for x in out.state.mass)
            assert all(x.dtype == jnp.float32 for x in out.state.number)
            assert all(x.dtype == jnp.float32 for x in out.properties.wet_radius)
            assert bool(jnp.all(out.valid))

            def _loss(gas_in):
                res = step_all_soa(
                    SoAState(soa_state.mass, soa_state.number, gas_in),
                    d['temperature'], d['pressure'], d['humidity'], jnp.float32(720.0),
                    d['production'], d['cloud'], table32, d['forest'], d['in_pbl'],
                    nucleation_scheme=scheme, organic_scheme=1,
                    ion_pair_rate=d['ipr'], kazil_lovejoy_table=kl,
                    compute_valid=False, compute_water=False, chunk_size=5,
                )
                return jnp.sum(res.state.gas * jnp.float32(1e-6) + res.state.number[0] * jnp.float32(1e-4))

            grad_gas = jax.jit(jax.grad(_loss))(soa_state.gas)
            assert grad_gas.dtype == jnp.float32
            assert bool(jnp.all(jnp.isfinite(grad_gas)))


def test_interface_soa_helpers_match_aos_and_validate_lengths():
    with _x64_mode(True):
        rng = np.random.default_rng(42)
        q_mass = rng.uniform(0.0, 1e-6, size=(3, 4, 18))
        q_num = rng.uniform(0.0, 1e8, size=(3, 4, 7))
        rho = rng.uniform(0.2, 1.3, size=(3, 4))

        nat_mass_aos = aerosol_mass_to_native(q_mass, rho[..., None])
        nat_mass_soa = aerosol_mass_to_native_soa(tuple(q_mass[..., i] for i in range(18)), rho)
        np.testing.assert_allclose(np.stack(nat_mass_soa, axis=-1), np.asarray(nat_mass_aos), rtol=5e-16, atol=0)

        back_mass_aos = native_to_aerosol_mass(nat_mass_aos, rho[..., None])
        back_mass_soa = native_to_aerosol_mass_soa(nat_mass_soa, rho)
        np.testing.assert_allclose(np.stack(back_mass_soa, axis=-1), np.asarray(back_mass_aos), rtol=5e-16, atol=0)

        nat_num_aos = number_mixing_ratio_to_native(q_num, rho[..., None])
        nat_num_soa = number_mixing_ratio_to_native_soa(tuple(q_num[..., i] for i in range(7)), rho)
        np.testing.assert_allclose(np.stack(nat_num_soa, axis=-1), np.asarray(nat_num_aos), rtol=5e-16, atol=0)

        back_num_aos = native_to_number_mixing_ratio(nat_num_aos, rho[..., None])
        back_num_soa = native_to_number_mixing_ratio_soa(nat_num_soa, rho)
        np.testing.assert_allclose(np.stack(back_num_soa, axis=-1), np.asarray(back_num_aos), rtol=5e-16, atol=0)

        with pytest.raises(ValueError, match='18'):
            aerosol_mass_to_native_soa((rho,) * 17, rho)
        with pytest.raises(ValueError, match='7'):
            number_mixing_ratio_to_native_soa((rho,) * 6, rho)
        assert 'chunk_size' in SOA_STATIC_OPTIONS
