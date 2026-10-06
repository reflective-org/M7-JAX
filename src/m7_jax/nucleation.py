"""M7 binary sulfuric-acid/water nucleation in native cm-based units.

Vehkamaeki polynomial terms, input clamps and rate/cluster limiters are copied
from mo_ham_m7_nucl.F90. M7's scheme 2 needs an external Kazil/Lovejoy lookup
table and is explicitly unsupported here. Standalone M7 compiles organic
activation/kinetic calls out; no SALSA growth/survival formula is substituted.
"""
from typing import NamedTuple

import jax.numpy as jnp

from .constants import EPS_DP


class NucleationRate(NamedTuple):
    rate: object
    cluster_sulfate: object


class NucleationResult(NamedTuple):
    gas: object
    number: object
    sulfate: object


def vehkamaeki(temperature_k, relative_humidity, gas_cm3):
    """Return rate (particles/cm3/s) and H2SO4 molecules per critical cluster.

    Temperature clamps to 190.15..300.15 K, RH to 1e-4..1, and gas to 1e4..1e11
    molecules/cm3. These are native parameterization clamps, including its
    evaluation at 1e4 when supplied vapor is zero. Rates below 1e-7 or total
    clusters below four molecules become zero; rates are capped at 1e10.
    """
    zt = jnp.clip(jnp.asarray(temperature_k), 190.15, 300.15)
    zrh = jnp.clip(jnp.asarray(relative_humidity), 1e-4, 1.)
    zrhoa = jnp.clip(jnp.asarray(gas_cm3), 1e4, 1e11)
    zt2 = zt * zt
    zt3 = zt2 * zt
    zlogrh = jnp.log(zrh)
    zlogrh2 = zlogrh * zlogrh
    zlogrh3 = zlogrh2 * zlogrh
    zlogrhoa = jnp.log(zrhoa)
    zlogrhoa2 = zlogrhoa * zlogrhoa
    zlogrhoa3 = zlogrhoa2 * zlogrhoa
    x = (0.7409967177282139 - 0.002663785665140117*zt
        + 0.002010478847383187*zlogrh
        - 0.0001832894131464668*zt*zlogrh
        + 0.001574072538464286*zlogrh2
        - 0.00001790589121766952*zt*zlogrh2
        + 0.0001844027436573778*zlogrh3
        -  1.503452308794887e-6*zt*zlogrh3
        - 0.003499978417957668*zlogrhoa
        + 0.0000504021689382576*zt*zlogrhoa)
    zix = 1. / x
    zjnuc = (0.1430901615568665 + 2.219563673425199*zt -
        0.02739106114964264*zt2 +
        0.00007228107239317088*zt3 + 5.91822263375044*zix +
        0.1174886643003278*zlogrh + 0.4625315047693772*zt*zlogrh -
        0.01180591129059253*zt2*zlogrh +
        0.0000404196487152575*zt3*zlogrh +
        (15.79628615047088*zlogrh)*zix -
        0.215553951893509*zlogrh2 -
        0.0810269192332194*zt*zlogrh2 +
        0.001435808434184642*zt2*zlogrh2 -
        4.775796947178588e-6*zt3*zlogrh2 -
        (2.912974063702185*zlogrh2)*zix -
        3.588557942822751*zlogrh3 +
        0.04950795302831703*zt*zlogrh3 -
        0.0002138195118737068*zt2*zlogrh3 +
        3.108005107949533e-7*zt3*zlogrh3 -
        (0.02933332747098296*zlogrh3)*zix +
        1.145983818561277*zlogrhoa -
        0.6007956227856778*zt*zlogrhoa +
        0.00864244733283759*zt2*zlogrhoa -
        0.00002289467254710888*zt3*zlogrhoa -
        (8.44984513869014*zlogrhoa)*zix +
        2.158548369286559*zlogrh*zlogrhoa +
        0.0808121412840917*zt*zlogrh*zlogrhoa -
        0.0004073815255395214*zt2*zlogrh*zlogrhoa -
        4.019572560156515e-7*zt3*zlogrh*zlogrhoa +
        (0.7213255852557236*zlogrh*zlogrhoa)*zix +
        1.62409850488771*zlogrh2*zlogrhoa -
        0.01601062035325362*zt*zlogrh2*zlogrhoa +
        0.00003771238979714162*zt2*zlogrh2*zlogrhoa +
        3.217942606371182e-8*zt3*zlogrh2*zlogrhoa -
        (0.01132550810022116*zlogrh2*zlogrhoa)*zix +
        9.71681713056504*zlogrhoa2 -
        0.1150478558347306*zt*zlogrhoa2 +
        0.0001570982486038294*zt2*zlogrhoa2 +
        4.009144680125015e-7*zt3*zlogrhoa2 +
        (0.7118597859976135*zlogrhoa2)*zix -
        1.056105824379897*zlogrh*zlogrhoa2 +
        0.00903377584628419*zt*zlogrh*zlogrhoa2 -
        0.00001984167387090606*zt2*zlogrh*zlogrhoa2 +
        2.460478196482179e-8*zt3*zlogrh*zlogrhoa2 -
        (0.05790872906645181*zlogrh*zlogrhoa2)*zix -
        0.1487119673397459*zlogrhoa3 +
        0.002835082097822667*zt*zlogrhoa3 -
        9.24618825471694e-6*zt2*zlogrhoa3 +
        5.004267665960894e-9*zt3*zlogrhoa3 -
        (0.01270805101481648*zlogrhoa3)*zix)
    zjnuc = jnp.exp(zjnuc)
    zntot = (-0.002954125078716302 - 0.0976834264241286*zt +
        0.001024847927067835*zt2 - 2.186459697726116e-6*zt3 -
        0.1017165718716887*zix - 0.002050640345231486*zlogrh -
        0.007585041382707174*zt*zlogrh +
        0.0001926539658089536*zt2*zlogrh -
        6.70429719683894e-7*zt3*zlogrh -
        (0.2557744774673163*zlogrh)*zix +
        0.003223076552477191*zlogrh2 +
        0.000852636632240633*zt*zlogrh2 -
        0.00001547571354871789*zt2*zlogrh2 +
        5.666608424980593e-8*zt3*zlogrh2 +
        (0.03384437400744206*zlogrh2)*zix +
        0.04743226764572505*zlogrh3 -
        0.0006251042204583412*zt*zlogrh3 +
        2.650663328519478e-6*zt2*zlogrh3 -
        3.674710848763778e-9*zt3*zlogrh3 -
        (0.0002672510825259393*zlogrh3)*zix -
        0.01252108546759328*zlogrhoa +
        0.005806550506277202*zt*zlogrhoa -
        0.0001016735312443444*zt2*zlogrhoa +
        2.881946187214505e-7*zt3*zlogrhoa +
        (0.0942243379396279*zlogrhoa)*zix -
        0.0385459592773097*zlogrh*zlogrhoa -
        0.0006723156277391984*zt*zlogrh*zlogrhoa +
        2.602884877659698e-6*zt2*zlogrh*zlogrhoa +
        1.194163699688297e-8*zt3*zlogrh*zlogrhoa -
        (0.00851515345806281*zlogrh*zlogrhoa)*zix -
        0.01837488495738111*zlogrh2*zlogrhoa +
        0.0001720723574407498*zt*zlogrh2*zlogrhoa -
        3.717657974086814e-7*zt2*zlogrh2*zlogrhoa -
        5.148746022615196e-10*zt3*zlogrh2*zlogrhoa +
        (0.0002686602132926594*zlogrh2*zlogrhoa)*zix -
        0.06199739728812199*zlogrhoa2 +
        0.000906958053583576*zt*zlogrhoa2 -
        9.11727926129757e-7*zt2*zlogrhoa2 -
        5.367963396508457e-9*zt3*zlogrhoa2 -
        (0.007742343393937707*zlogrhoa2)*zix +
        0.0121827103101659*zlogrh*zlogrhoa2 -
        0.0001066499571188091*zt*zlogrh*zlogrhoa2 +
        2.534598655067518e-7*zt2*zlogrh*zlogrhoa2 -
        3.635186504599571e-10*zt3*zlogrh*zlogrhoa2 +
        (0.0006100650851863252*zlogrh*zlogrhoa2)*zix +
        0.0003201836700403512*zlogrhoa3 -
        0.0000174761713262546*zt*zlogrhoa3 +
        6.065037668052182e-8*zt2*zlogrhoa3 -
        1.421771723004557e-11*zt3*zlogrhoa3 +
        (0.0001357509859501723*zlogrhoa3)*zix)
    zntot = jnp.exp(zntot)
    rate = jnp.where((zjnuc < 1e-7) | (zntot < 4.), 0., zjnuc)
    rate = jnp.minimum(rate, 1e10)
    return NucleationRate(rate, zntot * x)


def organic_activation(h2so4_clear, forest, in_pbl):
    """Native ``nucl_activation`` (mo_ham_m7_nucl.F90:672-736).

    Activation-type organic nucleation (Kulmala et al., ACP 2006), active
    only over forests and only in the boundary layer. ``in_pbl`` is a
    per-cell boolean the host precomputes as the Fortran's own level test,
    ``jk >= int(ppbl)`` with a 1-based, top-first ``jk``; this routine has no
    notion of levels itself. Returns the formation rate [cm-3 s-1] and the
    number of H2SO4 molecules in each new particle (always 1 here).
    """
    # Dtype-generic (#1017 W1 task 3): anchor on h2so4_clear, the carrier the
    # step_all chain passes in whatever working dtype it is running.
    h2so4_clear = jnp.asarray(h2so4_clear)
    forest = jnp.asarray(forest, dtype=h2so4_clear.dtype)
    in_pbl = jnp.asarray(in_pbl)
    rate = jnp.where(in_pbl, 1.0e-6 * h2so4_clear * forest, 0.0)
    return NucleationRate(rate, jnp.where(in_pbl, 1.0, 0.0))


def organic_kinetic(h2so4_clear, forest, in_pbl):
    """Native ``nucl_kinetic`` (mo_ham_m7_nucl.F90:740-808).

    Kinetic-type organic nucleation (Laakso et al., ACP 2004), same forest/
    boundary-layer gating as :func:`organic_activation`. Each new particle
    carries 2 H2SO4 molecules here.
    """
    h2so4_clear = jnp.asarray(h2so4_clear)
    forest = jnp.asarray(forest, dtype=h2so4_clear.dtype)
    in_pbl = jnp.asarray(in_pbl)
    rate = jnp.where(in_pbl, 3.86e-13 * h2so4_clear * h2so4_clear * forest, 0.0)
    return NucleationRate(rate, jnp.where(in_pbl, 2.0, 0.0))


def nucleate_all(gas_cm3, temperature_k, relative_humidity, sink, cloud_fraction,
                  forest, in_pbl, timestep, *, scheme=1, organic_scheme=0,
                  initial_number=0., initial_sulfate=0.):
    """Port standalone ``m7_nuck`` (mo_ham_m7.F90:2257-2465), combining
    sulfuric-acid nucleation (``scheme``, as in :func:`nucleate`) with
    organic nucleation (``organic_scheme``: 0 off, 1 activation, 2 kinetic)
    exactly as the native routine does -- summed rates, combined H2SO4 loss,
    the same cloud-free scaling and the ``zeps`` activity gate -- so that the
    number/mass increments come out identical whichever mechanism (or both)
    is active. ``scheme`` and ``organic_scheme`` are static JIT arguments.
    """
    if scheme == 2:
        raise NotImplementedError('M7 Kazil/Lovejoy nucleation requires its external lookup table')
    if scheme not in (0, 1):
        raise ValueError('Supported M7 sulfur nucleation schemes are 0 and 1')
    if organic_scheme not in (0, 1, 2):
        raise ValueError('Supported M7 organic nucleation schemes are 0, 1 and 2')
    gas, cloud, sink = jnp.asarray(gas_cm3), jnp.asarray(cloud_fraction), jnp.asarray(sink)
    clear_gas = jnp.where(cloud >= 1., 0.,
        jnp.where(cloud <= 0., gas, gas / jnp.where(cloud < 1., 1. - cloud, 1.)))
    if scheme == 1:
        snrate, sns = vehkamaeki(temperature_k, relative_humidity, clear_gas)
    else:
        snrate = sns = jnp.zeros_like(clear_gas)
    if organic_scheme == 1:
        onrate, ons = organic_activation(clear_gas, forest, in_pbl)
    elif organic_scheme == 2:
        onrate, ons = organic_kinetic(clear_gas, forest, in_pbl)
    else:
        onrate = ons = jnp.zeros_like(clear_gas)
    total_rate = snrate + onrate
    active = total_rate > EPS_DP
    loss = sns * snrate + ons * onrate
    clear_final = clear_gas - timestep * loss / (1. + timestep * sink)
    clear_final = jnp.minimum(jnp.maximum(clear_final, 0.), clear_gas)
    delta = (clear_gas - clear_final) * (1. - cloud)
    number = delta * total_rate / jnp.where(active, loss, 1.)
    return NucleationResult(jnp.where(active, gas - delta, gas),
        jnp.where(active, number, initial_number),
        jnp.where(active, delta, initial_sulfate))


def nucleate(gas_cm3, temperature_k, relative_humidity, sink, cloud_fraction,
             timestep, *, scheme=1, initial_number=0., initial_sulfate=0.):
    """Port standalone m7_nuck, returning gas and new-particle/sulfate deltas.

    Inputs gas/sulfate are molecules/cm3, number particles/cm3, sink s-1 and
    timestep seconds. The scheme is a static JIT argument (0 disabled, 1 binary).
    Inactive rates preserve the caller's initial delta values, matching native
    INOUT behavior. Standard composition initializes both deltas to zero.
    """
    if scheme == 2:
        raise NotImplementedError('M7 Kazil/Lovejoy nucleation requires its external lookup table')
    if scheme not in (0, 1):
        raise ValueError('Supported M7 sulfur nucleation schemes are 0 and 1')
    gas, cloud, sink = jnp.asarray(gas_cm3), jnp.asarray(cloud_fraction), jnp.asarray(sink)
    if scheme == 0:
        shape = jnp.broadcast_shapes(gas.shape, cloud.shape, sink.shape)
        return NucleationResult(jnp.broadcast_to(gas, shape),
                                jnp.broadcast_to(jnp.asarray(initial_number), shape),
                                jnp.broadcast_to(jnp.asarray(initial_sulfate), shape))
    clear_gas = jnp.where(cloud >= 1., 0.,
        jnp.where(cloud <= 0., gas, gas / jnp.where(cloud < 1., 1. - cloud, 1.)))
    rate, cluster = vehkamaeki(temperature_k, relative_humidity, clear_gas)
    active = rate > EPS_DP
    loss = cluster * rate
    clear_final = clear_gas - timestep * loss / (1. + timestep * sink)
    clear_final = jnp.minimum(jnp.maximum(clear_final, 0.), clear_gas)
    delta = (clear_gas - clear_final) * (1. - cloud)
    number = delta * rate / jnp.where(active, loss, 1.)
    return NucleationResult(jnp.where(active, gas - delta, gas),
        jnp.where(active, number, initial_number),
        jnp.where(active, delta, initial_sulfate))
