"""Multi-species M7 coagulation (``dnum_all``) parity against the actual

Fortran ``m7_delcoa`` (both sections). Designed cases cover populated and
empty insoluble modes, zero and nonzero mass_delta seeds in every allowed
species/mode slot (exercising the native "=" vs "+=" INOUT semantics),
nucleation into a populated and an empty NS mode, and coagulation on/off.
"""
import jax
import numpy as np
import pytest

from m7_jax.coagulation import dnum_all
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update("jax_enable_x64", True)

NATIVE_SO4 = 98.08 / 6.022e23  # g per molecule, matching the native cases() helpers elsewhere


def cases():
    n = 10
    number = np.zeros((n, 7))
    # NS KS AS CS; case 1 starts NS empty to exercise nucleation into an
    # empty mode; case 7 has essentially no insoluble population at all.
    number[:, 0] = [2e4, 0, 1e4, 5e3, 2e4, 2e4, 2e4, 1e-12, 2e4, 2e4]
    number[:, 1] = [300, 2e4, 200, 100, 300, 300, 300, 1e-12, 300, 300]
    number[:, 2] = [30, 10, 30, 50, 30, 30, 30, 1e-12, 30, 30]
    number[:, 3] = [0.2, 0.1, 1, 2, 0.2, 0.2, 0.2, 1e-12, 0.2, 0.2]
    number[:, 4] = [50, 50, 0, 50, 50, 1e-12, 50, 1e-12, 50, 50]      # KI
    number[:, 5] = [5, 5, 5, 0, 5, 5, 1e-12, 1e-12, 5, 5]              # AI
    number[:, 6] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-12, 1e-12, 0.5]  # CI

    radius = np.tile([2e-7, 2e-6, 1e-5, 1e-4, 3e-6, 2e-5, 1.5e-4], (n, 1))
    density = np.tile([1.841, 1.7, 1.4, 1.2, 1.8, 2.6, 2.6], (n, 1))

    mass = np.zeros((n, 18))
    mass[:, 0] = number[:, 0] * (4 * np.pi / 3) * radius[:, 0] ** 3 * density[:, 0] / NATIVE_SO4
    mass[:, 1] = number[:, 1] * (4 * np.pi / 3) * radius[:, 1] ** 3 * density[:, 1] / NATIVE_SO4
    mass[:, 2] = number[:, 2] * (4 * np.pi / 3) * radius[:, 2] ** 3 * density[:, 2] / NATIVE_SO4
    mass[:, 3] = number[:, 3] * (4 * np.pi / 3) * radius[:, 3] ** 3 * density[:, 3] / NATIVE_SO4
    # Every allowed species/mode slot gets a distinct per-particle mass so
    # mismatched indices show up as wrong magnitudes, not accidental matches.
    mass[:, 4] = number[:, 1] * 1e-15 * 1e12   # BC KS
    mass[:, 5] = number[:, 2] * 1.3e-15 * 1e12  # BC AS
    mass[:, 6] = number[:, 3] * 1.6e-15 * 1e12  # BC CS
    mass[:, 7] = number[:, 4] * 1.9e-15 * 1e12  # BC KI
    mass[:, 8] = number[:, 1] * 2e-15 * 1e12    # OC KS
    mass[:, 9] = number[:, 2] * 2.3e-15 * 1e12  # OC AS
    mass[:, 10] = number[:, 3] * 2.6e-15 * 1e12  # OC CS
    mass[:, 11] = number[:, 4] * 2.9e-15 * 1e12  # OC KI
    mass[:, 12] = number[:, 2] * 3e-15 * 1e12   # SS AS
    mass[:, 13] = number[:, 3] * 3.3e-15 * 1e12  # SS CS
    mass[:, 14] = number[:, 2] * 4e-15 * 1e12   # DU AS
    mass[:, 15] = number[:, 3] * 4.3e-15 * 1e12  # DU CS
    mass[:, 16] = number[:, 5] * 4.6e-15 * 1e12  # DU AI
    mass[:, 17] = number[:, 6] * 4.9e-15 * 1e12  # DU CI

    nucleated_number = np.array([30000., 30000, 0, 30000, 30000, 30000, 30000, 0, 30000, 30000])
    mass_delta = np.zeros((n, 18))
    mass_delta[:, 0] = nucleated_number * 30
    # Seed every other slot nonzero too: the overwritten targets (iso4ks/as/cs
    # and the section-1 ibcks/iocks/ibcas/iocas) must discard this seed, while
    # the add-only targets (ibccs/ioccs, iduas/iducs, ibcki/iocki/iduai/iduci
    # pass-throughs) must retain and accumulate it.
    mass_delta[:, 1:] += 7.0

    radius_dry = radius[:, :4].copy()
    insoluble_sulfate = np.zeros((n, 3))

    timestep = np.full(n, 600.)
    temperature = np.linspace(230, 310, n)
    pressure = np.linspace(3e4, 101325, n)
    coagulation = np.ones(n)
    coagulation[-1] = 0.

    return dict(number=number, radius=radius, density=density, mass=mass, mass_delta=mass_delta,
                dry_radius=radius_dry, insoluble_sulfate=insoluble_sulfate, temperature=temperature,
                pressure=pressure, timestep=timestep, nucleated_number=nucleated_number,
                coagulation=coagulation)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    executable, _ = build_oracle(tmp_path_factory.mktemp("m7-coagulation-multispecies"), checked=True)
    return executable


def invoke(data, mean_mass, *, compiled=False):
    n = data["number"].shape[0]
    function = dnum_all
    if compiled:
        function = jax.jit(function, static_argnames=("coagulation",))
    results = [function(data["mass"][i], data["number"][i], data["temperature"][i], data["pressure"][i],
                         data["radius"][i], data["density"][i], data["insoluble_sulfate"][i],
                         data["nucleated_number"][i], data["mass_delta"][i], mean_mass[i], data["timestep"][i],
                         coagulation=bool(data["coagulation"][i]))
               for i in range(n)]
    mass = np.array([np.asarray(r[0]) for r in results])
    number = np.array([np.asarray(r[1]) for r in results])
    delta = np.array([np.asarray(r[2]) for r in results])
    return mass, number, delta


@pytest.mark.parametrize("compiled", [False, True])
def test_dnum_all_native_multispecies_ordered_transfers(native, compiled):
    data = cases()
    # pttn/mean_particle_mass is an input to dnum_all, exactly as to the
    # native m7_dnum; it is computed once from the same mass/number/radius
    # that both the oracle's 'dnum' call and dnum_all itself receive.
    props, _ = run_oracle(native, data, operation="properties")
    mean_mass = props["mean_particle_mass"]
    expected, _ = run_oracle(native, data, operation="dnum")

    mass, number, delta = invoke(data, mean_mass, compiled=compiled)

    # The fitted analytical decay (shared with the sulfate-only module) and
    # the chained insoluble/soluble corrections amplify roundoff a little
    # beyond the pure-sulfate module's ~1e-13; still far inside the task's
    # 1e-10 "not attributable to operation order" STOP threshold.
    np.testing.assert_allclose(mass, expected["mass"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(number, expected["number"], rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(delta[..., :4], expected["sulfate_delta"], rtol=1e-11, atol=1e-15)
    assert np.isfinite(mass).all() and np.isfinite(number).all()
    assert (mass >= -1e-20).all() and (number >= -1e-20).all()
    # Sea salt has no insoluble companion mode: its soluble-mode mass must be
    # completely untouched by modal coagulation (mo_ham_m7.F90:4654-4702 has
    # no id_ss branch in the final aggregation).
    np.testing.assert_array_equal(mass[..., 12:14], data["mass"][..., 12:14])


def test_dnum_all_conserves_species_mass_and_sulfur(native):
    # Realistic INOUT usage: mass_delta carries only the nucleated sulfate
    # seed (as m7_nuck would hand it in). cases()'s deliberate non-nucleation
    # seeding of every mass_delta slot is for test_dnum_all_native_... above,
    # which validates the native overwrite-vs-accumulate semantics for those
    # slots directly against the oracle; it is not a conservation scenario
    # (two of BC's four pa4delt slots are native OVERWRITE targets, mo_ham_m7
    # .F90:3756-3759/3801-3803, so an arbitrary external seed there is simply
    # discarded by design, not conserved).
    data = cases()
    data["mass_delta"] = np.zeros_like(data["mass_delta"])
    data["mass_delta"][:, 0] = data["nucleated_number"] * 30
    props, _ = run_oracle(native, data, operation="properties")
    mean_mass = props["mean_particle_mass"]
    mass, number, delta = invoke(data, mean_mass, compiled=False)

    for name, idx in (("BC", [4, 5, 6, 7]), ("OC", [8, 9, 10, 11]), ("DU", [14, 15, 16, 17])):
        before = data["mass"][..., idx].sum(-1)
        after = mass[..., idx].sum(-1)
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-10,
                                    err_msg=f"{name} mass not conserved")
    # Sulfate (plus the externally nucleated input) is conserved exactly;
    # there is no insoluble condensate flux in this designed set.
    so4_before = data["mass"][..., :4].sum(-1)
    so4_after = mass[..., :4].sum(-1)
    budget = so4_after - so4_before - data["mass_delta"][..., 0]
    np.testing.assert_allclose(budget, 0., atol=1e-3)  # ~1e10-molecule-scale values; roundoff only


def test_invalid_shapes_rejected():
    data = cases()
    mean_mass = np.ones((10, 18))
    with pytest.raises(ValueError, match="dimension 18"):
        dnum_all(np.ones(7), data["number"][0], 280., 1e5, data["radius"][0], data["density"][0],
                 np.zeros(3), 0., np.zeros(18), mean_mass[0], 600.)
    with pytest.raises(ValueError, match="dimension 3"):
        dnum_all(data["mass"][0], data["number"][0], 280., 1e5, data["radius"][0], data["density"][0],
                 np.zeros(7), 0., np.zeros(18), mean_mass[0], 600.)
